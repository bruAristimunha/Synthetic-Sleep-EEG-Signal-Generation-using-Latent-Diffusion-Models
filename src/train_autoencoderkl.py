"""
Author: Bruno Aristimunha
Training AutoEncoder KL with SleepEDFx or SHHS data.
Based on the tutorial from Monai Generative.

"""
import argparse
import ast
import time
import os
from generative.losses import JukeboxLoss, PatchAdversarialLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator
from monai.config import print_config
from monai.utils import first, set_determinism
from omegaconf import OmegaConf
from torch.nn import L1Loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.dataset import train_dataloader, valid_dataloader, get_trans
from util import log_mlflow, log_reconstructions, log_spectral, ParseListAction, print_gpu_memory_report, setup_run_dir

# print_config()

# for reproducibility purposes set a seed
set_determinism(42)

import torch

if os.path.exists('/project'):
    base_path = '/project/'
    base_path_data = '/data/'
else:
    base_path = '/home/bru/PycharmProjects/DDPM-EEG/'
    base_path_data = base_path

class ParseListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parsed_list = ast.literal_eval(values)
        setattr(namespace, self.dest, parsed_list)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        #default="/home/bru/PycharmProjects/DDPM-EEG/config/config_aekl_eeg.yaml",
        default="/project/config/config_encoder_eeg.yaml",
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
        type=str, action=ParseListAction
    )

    parser.add_argument(
        "--spe",
        type=str, 
    )
    parser.add_argument(
        "--latent_channels",
        type=int,
    )
    parser.add_argument(
        "--type_dataset",
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

    run_dir, resume = setup_run_dir(config=config, args=args, base_path=base_path)

    # Getting write training and validation data

    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))

    trans = get_trans(args.dataset)
    # Getting data loaders
    train_loader = train_dataloader(config=config, args=args, transforms_list=trans, dataset=args.dataset)
    val_loader = valid_dataloader(config=config, args=args, transforms_list=trans, dataset=args.dataset)

    # Defining device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Defining model
    autoencoder_args = config.autoencoderkl.params
    autoencoder_args['num_channels'] = args.num_channels
    autoencoder_args['latent_channels'] = args.latent_channels

    model = AutoencoderKL(**autoencoder_args)
    # including extra parameters for the discriminator from a dictionary
    discriminator_dict = config.patchdiscriminator.params

    discriminator = PatchDiscriminator(**discriminator_dict)

    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.device_count() > 1:
        print("Putting the model to run in more that 1 GPU")
        model = torch.nn.DataParallel(model)
        discriminator = torch.nn.DataParallel(discriminator)

    model = model.to(device)
    discriminator = discriminator.to(device)

    optimizer_g = torch.optim.Adam(params=model.parameters(),
                                   lr=config.models.optimizer_g_lr)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(),
                                   lr=config.models.optimizer_d_lr)

    # %%
    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_weight = config.models.adv_weight
    jukebox_loss = JukeboxLoss(spatial_dims=1, reduction="sum")

    # ## Model Training

    kl_weight = config.models.kl_weight
    n_epochs = config.train.n_epochs
    val_interval = config.train.val_interval
    spectral_weight = config.models.spectral_weight
    epoch_recon_loss_list = []
    epoch_gen_loss_list = []
    epoch_disc_loss_list = []
    epoch_spectral_loss_list = []
    val_recon_epoch_loss_list = []
    best_loss = float("inf")
    start_epoch = 0
    if resume:
        print(f"Using checkpoint!")
        checkpoint = torch.load(str(run_dir / "checkpoint.pth"))
        model.load_state_dict(checkpoint["state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        init_batch = checkpoint["init_batch"]
    else:
        print(f"No checkpoint found.")

    total_start = time.time()

    init_batch = first(train_loader)['eeg'].to(device)[:,:,36:-36]

    for epoch in range(start_epoch, n_epochs):
        model.train()
        discriminator.train()
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        spectral_epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=150)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in progress_bar:
            eeg_data = batch['eeg'].to(device)

            optimizer_g.zero_grad(set_to_none=True)
            reconstruction, z_mu, z_sigma = model(eeg_data)

            recons_loss = l1_loss(reconstruction.float(), eeg_data.float())

            recons_spectral = jukebox_loss(reconstruction.float(), eeg_data.float())

            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            if args.spe == "spectral":
                loss_g = recons_loss + kl_weight * kl_loss + adv_weight * generator_loss  + recons_spectral * spectral_weight
            else:
                loss_g = recons_loss + kl_weight * kl_loss + adv_weight * generator_loss 
            loss_g.backward()
            optimizer_g.step()

            # Discriminator part
            optimizer_d.zero_grad(set_to_none=True)

            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = discriminator(eeg_data.contiguous().detach())[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            loss_d = adv_weight * discriminator_loss

            loss_d.backward()
            optimizer_d.step()

            epoch_loss += recons_loss.item()
            gen_epoch_loss += generator_loss.item()
            disc_epoch_loss += discriminator_loss.item()
            spectral_epoch_loss += recons_spectral.item()

            progress_bar.set_postfix(
                {
                    "l1_loss": epoch_loss / (step + 1),  # L1 loss
                    "gen_loss": gen_epoch_loss / (step + 1),  # Generator loss, adv_loss
                    "disc_loss": disc_epoch_loss / (step + 1),  # Discriminator loss, adv_loss
                    "spec_loss": spectral_epoch_loss / (step + 1),  # Spectral loss, jukebox_loss
                }
            )

            if (epoch + 1) % val_interval == 0:
                with torch.no_grad():
                    log_reconstructions(img=eeg_data[:,:,36:-36],
                                        recons=reconstruction[:,:,36:-36],
                                        writer=writer_train,
                                        step=epoch+1,
                                        name="RECONSTRUCTION_TRAIN",
                                        run_dir=run_dir)
        
                    reconstruction_init, _, _ = model(init_batch)

                    log_reconstructions(img=init_batch[:,:,36:-36],
                                        recons=reconstruction_init[:,:,36:-36],
                                        writer=writer_train,
                                        step=epoch+1,
                                        name="RECONSTRUCTION_TRAIN_OVERTIME",
                                        run_dir=run_dir)

                    log_spectral(eeg=eeg_data[:,:,36:-36],
                                 recons=reconstruction_init[:,:,36:-36],
                                 writer=writer_val,
                                 step=epoch+1,
                                 name="SPECTROGRAM_OVERTIME", 
                                 run_dir=run_dir)

        writer_train.add_scalar("loss_g", gen_epoch_loss / (step + 1), epoch * len(train_loader) + step)
        writer_train.add_scalar("loss_d", disc_epoch_loss / (step + 1), epoch * len(train_loader) + step)
        writer_train.add_scalar("recons_loss", epoch_loss / (step + 1), epoch * len(train_loader) + step)
        writer_train.add_scalar("recons_spectral", spectral_epoch_loss / (step + 1), epoch * len(train_loader) + step)

        epoch_recon_loss_list.append(epoch_loss / (step + 1))
        epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
        epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))
        epoch_spectral_loss_list.append(disc_epoch_loss / (step + 1))

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader, start=1):
                    eeg_data = batch['eeg'].to(device)
                    reconstruction_eeg, _, _ = model(eeg_data)


                    log_reconstructions(img=eeg_data[:,:,36:-36],
                                        recons=reconstruction_eeg[:,:,36:-36],
                                        writer=writer_val,
                                        step=epoch+1,
                                        name="RECONSTRUCTION_VAL",
                                        run_dir=run_dir)

                    log_spectral(eeg=eeg_data[:,:,36:-36],
                                 recons=reconstruction_eeg[:,:,36:-36],
                                 writer=writer_val,
                                 step=epoch+1,
                                 name="SPECTROGRAM_VAL", 
                                 run_dir=run_dir)

                    recons_loss = l1_loss(reconstruction_eeg.float(),
                                          eeg_data.float())

                    val_loss += recons_loss.item()

                    if val_loss <= best_loss:
                        print(f"New best val loss {val_loss}")
                        best_loss = val_loss
                        torch.save(model.state_dict(), str(run_dir / "best_model.pth"))

                    print_gpu_memory_report()
                    # Save checkpoint
                    checkpoint = {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "optimizer_g": optimizer_g.state_dict(),
                        "optimizer_d": optimizer_d.state_dict(),
                        "best_loss": best_loss,
                        "init_batch": init_batch,
                    }
                    torch.save(checkpoint, str(run_dir / "checkpoint.pth"))

            val_loss /= val_step
            writer_val.add_scalar("recons_loss", val_loss, epoch * len(val_loader) + step)

            val_recon_epoch_loss_list.append(val_loss)

    total_time = time.time() - total_start
    print(f"train completed, total time: {total_time}.")
    torch.save(model.state_dict(), str(run_dir / "final_model.pth"))

    log_mlflow(
        model=model,
        config=config,
        args=args,
        run_dir=run_dir,
        val_loss=val_loss,
    )
    #wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
