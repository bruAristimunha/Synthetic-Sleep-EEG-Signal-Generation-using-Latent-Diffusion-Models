"""
Author: Bruno Aristimunha
Training LDM with SleepEDFx or SHHS data.
Based on the tutorial from Monai Generative.

"""
import argparse

import torch
import torch.nn as nn

from generative.networks.nets import DiffusionModelUNet
from generative.networks.nets import AutoencoderKL
from generative.networks.schedulers import DDPMScheduler
from monai.config import print_config
from monai.utils import first, set_determinism
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter


from dataset.dataset import train_dataloader, valid_dataloader
from models.ldm import UNetModel
from training import train_ldm
from util import log_mlflow, ParseListAction, setup_run_dir

# print_config()
# for reproducibility purposes set a seed

set_determinism(42)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        #default="/home/bru/PycharmProjects/DDPM-EEG/config/config_ldm.yaml",
        default="/project/config/config_ldm.yaml",
        help="Path to config file with all the training parameters needed",
    )
    parser.add_argument(
        "--path_train_ids",
        type=str,
        #default="/home/bru/PycharmProjects/DDPM-EEG/data/ids/ids_sleep_edfx_cassette_train.csv",
        default="/project/data/ids/ids_sleep_edfx_cassette_double_train.csv",
    )

    parser.add_argument(
        "--path_valid_ids",
        type=str,
        #default="/home/bru/PycharmProjects/DDPM-EEG/data/ids/ids_sleep_edfx_cassette_valid.csv",
        default="/project/data/ids/ids_sleep_edfx_cassette_double_valid.csv",
    )
    parser.add_argument(
        "--path_cached_data",
        type=str,
        #default="/home/bru/PycharmProjects/DDPM-EEG/data/pre",
        default="/data/pre",
    )

    parser.add_argument(
        "--path_pre_processed",
        type=str,
        #default="/home/bru/PycharmProjects/DDPM-EEG/data/pre-processed",
        default="/data/physionet-sleep-data-npy",
    )
    parser.add_argument(
        "--num_channels",
        type=str, action=ParseListAction,
        )
    parser.add_argument("--autoencoderkl_config_file_path",
                        help="Path to the .pth model from the stage1.",
                        default="/project/config/config_aekl_eeg.yaml")
                        #default="/home/bru/PycharmProjects/DDPM-EEG/config/config_aekl_eeg.yaml")

    parser.add_argument("--best_model_path",
                        help="Path to the .pth model from the stage1.",)
                        #default="/home/bru/PycharmProjects/DDPM-EEG/aekl_eeg_may_30_3/aekl_eeg_32-32-64_spectral_4h/best_model.pth")
    parser.add_argument(
        "--spe",
        type=str,
    )
    parser.add_argument(
        "--latent_channels",
        type=int,
    )

    args = parser.parse_args()
    return args


class Stage1Wrapper(nn.Module):
    """Wrapper for stage 1 model as a workaround for
    the DataParallel usage in the training loop."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_mu, z_sigma = self.model.encode(x)
        z = self.model.sampling(z_mu, z_sigma)
        return z

def main(args):
    config = OmegaConf.load(args.config_file)

    set_determinism(seed=config.train.seed)
    print_config()

    run_dir, resume = setup_run_dir(config=config, args=args)

    # Getting write training and validation data

    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))

    # Getting data loaders
    train_loader = train_dataloader(config=config, args=args)
    val_loader = valid_dataloader(config=config, args=args)
    # Defining device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Defining model
    config_aekl = OmegaConf.load(args.autoencoderkl_config_file_path)
    autoencoder_args = config_aekl.autoencoderkl.params
    autoencoder_args['num_channels'] = args.num_channels
    autoencoder_args['latent_channels'] = args.latent_channels

    stage1 = AutoencoderKL(**autoencoder_args)

    state_dict = torch.load(args.best_model_path+"/best_model.pth",
                            map_location=torch.device('cpu'))

    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v

    stage1.load_state_dict(state_dict)
    stage1.to(device)
    with torch.no_grad():
        with autocast(enabled=True):
            check_data = first(train_loader)['eeg'].to(device)
            z = stage1.encode_stage_2_inputs(check_data)

    autoencoderkl = Stage1Wrapper(model=stage1)

    #########################################################################
    # Diffusion model part
    # spatial_dims: 1
    # in_channels: 1
    # out_channels: 1
    # num_channels: [1, 2, 4]
    # latent_channels: 1
    # num_res_blocks: 2
    # norm_num_groups: 1
    # attention_levels: [false, false, false]
    # with_encoder_nonlocal_attn: false
    # with_decoder_nonlocal_attn: false
    #
    # diffusion = DiffusionModelUNet(
    #     spatial_dims=1,
    #     in_channels=1,
    #     out_channels=1,
    #     num_res_blocks=[8,4],
    #     num_channels=[1,2],
    #     attention_levels=(False, False),
    #     norm_num_groups=1,
    #     norm_eps=1e-6,
    #     resblock_updown=False,
    #     num_head_channels=1,
    #     with_conditioning=False,
    #     transformer_num_layers=1,
    #     cross_attention_dim=None,
    #     num_class_embeds=None,
    #     upcast_attention=False,
    #     use_flash_attention=False,
    # )
    #print(diffusion)
    parameters = config['model']['params']['unet_config']['params']
    parameters['in_channels'] = args.latent_channels
    parameters['out_channels'] = args.latent_channels

    diffusion = UNetModel(**parameters)

    if torch.cuda.device_count() > 1:
        autoencoderkl = torch.nn.DataParallel(autoencoderkl)
        diffusion = torch.nn.DataParallel(diffusion)

    autoencoderkl.eval()
    
    autoencoderkl = autoencoderkl.to(device)
    diffusion.to(device)

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear",
                              beta_start=0.0015, beta_end=0.0195)
    
    scheduler.to(device)
    print(f"Scaling factor set to {1 / torch.std(z)}")
    scale_factor = 1 / torch.std(z)

    #inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)

    best_loss = float("inf")
    start_epoch = 0

    print(f"Starting Training")
    val_loss = train_ldm(
        model=diffusion,
        stage1=autoencoderkl,
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
        scale_factor=scale_factor,
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
