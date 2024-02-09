"""
Author: Bruno Aristimunha
Training LDM with SleepEDFx or SHHS data.
Based on the tutorial from Monai Generative.

"""
import argparse
import os
import torch
import torch.nn as nn

from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from monai.config import print_config
from monai.utils import first, set_determinism
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from generative.inferers import DiffusionInferer


from dataset.dataset import train_dataloader, valid_dataloader, get_trans
from models.ldm import UNetModel
from training.training_diffusion import train_diffusion_
from util import log_mlflow, ParseListAction, setup_run_dir
from generative.networks.nets import DiffusionModelUNet
# print_config()
# for reproducibility purposes set a seed

set_determinism(42)

if os.path.exists('/project'):
    base_path = '/project/'
    base_path_data = '/data/'
else:
    base_path = '/home/bru/PycharmProjects/DDPM-EEG/'
    base_path_data = base_path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=f"{base_path}/config/config_dm.yaml",
        help="Path to config file with all the training parameters needed",
    )
    parser.add_argument(
        "--path_train_ids",
        type=str,
        default=f"{base_path}/data/ids/ids_sleep_edfx_cassette_double_train.csv",
    )

    parser.add_argument(
        "--path_valid_ids",
        type=str,
        default=f"{base_path}/data/ids/ids_sleep_edfx_cassette_double_valid.csv",
    )
    parser.add_argument(
        "--path_cached_data",
        type=str,
        default=f"{base_path_data}/pre",
    )

    parser.add_argument(
        "--path_pre_processed",
        type=str,
        #default="/home/bru/PycharmProjects/DDPM-EEG/data/data_test",
        default="/data/physionet-sleep-data-npy",
    )
        
    parser.add_argument(
        "--spe",
        type=str,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["edfx", "shhs", "shhsh"]
    )

    args = parser.parse_args()
    return args



def main(args):
    config = OmegaConf.load(args.config_file)

    set_determinism(seed=config.train.seed)
    print_config()

    run_dir, resume = setup_run_dir(config=config, args=args,
                                    base_path=base_path)

    # Getting write training and validation data

    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))
    trans = get_trans(args.dataset)
    # Getting data loaders
    train_loader = train_dataloader(config=config, args=args, transforms_list=trans, dataset=args.dataset)
    val_loader = valid_dataloader(config=config, args=args, transforms_list=trans, dataset=args.dataset)

    first(train_loader)

    # Defining device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    parameters = config['model']['params']['unet_config']['params']
    parameters['in_channels'] = 1
    parameters['out_channels'] = 1

    diffusion = UNetModel(**parameters)
    
    if torch.cuda.device_count() > 1:
        diffusion = torch.nn.DataParallel(diffusion)

    diffusion.to(device)

    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta",
                              beta_start=0.0015, beta_end=0.0195)
    
    scheduler.to(device)
    if args.spe == 'spectral':
        spectral_loss = True
        print("using spectral loss")
    else:
        spectral_loss = False

    inferer = DiffusionInferer(scheduler)

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)

    best_loss = float("inf")
    start_epoch = 0

    print(f"Starting Training")
    val_loss = train_diffusion_(
        model=diffusion,
        scheduler=scheduler,
        start_epoch=start_epoch,
        best_loss=best_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=config.train.n_epochs,
        eval_freq=config.train.eval_freq,
        writer_train=writer_train,
        writer_val=writer_val,
        device=device,
        run_dir=run_dir,
        inferer=inferer,
        spectral_loss=spectral_loss,
        spectral_weight=1E-6
    )

    log_mlflow(
        model=diffusion,
        config=config,
        args=args,
        run_dir=run_dir,
        val_loss=val_loss,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
