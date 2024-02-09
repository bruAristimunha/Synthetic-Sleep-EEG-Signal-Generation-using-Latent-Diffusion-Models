"""
 Training Chambon Net with SleepEDFx data
 """

import argparse
import shutil
from collections import OrderedDict
from pathlib import Path

import lpips
import mne
import torch
import torch.nn.functional as F
import torch.optim as optim
from mne.datasets.sleep_physionet.age import fetch_data
from monai.config import print_config
from monai.data import Dataset
from monai.transforms import Compose, LoadImageD, SpatialCropD
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.ae_kl import AutoencoderKL
from models.discriminator import Discriminator, weights_init
from util import log_mlflow, log_reconstructions


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default="config/config.yaml",
        help="Path to config file with all the training parameters needed",
    )

    args = parser.parse_args()
    return args


def getting_data(base_path, subject_ids=None, recording_ids=None):
    """
    Get data from mne.datasets.sleep_physionet.age.fetch_data
    :param subject_ids:
    :param recording_ids:
    :return: list of paths to npy files.
    """
    #mne.set_config("PHYSIONET_SLEEP_PATH", base_path)

    if subject_ids is None:
        subject_ids = range(40)
    if recording_ids is None:
        recording_ids = [1, 2]

    paths = fetch_data(
        subjects=subject_ids,
        recording=recording_ids, on_missing='warn')
    eeg_signal = []
    for p in paths:
        file_save = p[0].replace(".edf", ".npy")
        eeg_signal.append({'eeg': file_save})

    return eeg_signal


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_dataloader(root, config):
    file_list = getting_data(root, subject_ids=[1, 2, 3], recording_ids=[1])
    sfreq = 100

    eeg_transforms = Compose([LoadImageD(keys='eeg'),
                              SpatialCropD(keys='eeg',
                                           roi_slices=[slice(29952),
                                                       slice(29952)])])

    dataset = Dataset(data=file_list, transform=eeg_transforms)

    loader = DataLoader(
        dataset=dataset,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        shuffle=True,
    )
    return loader


def test_dataloader(root, config):
    file_list = getting_data(root, subject_ids=[5, 6], recording_ids=[1])
    sfreq = 100

    eeg_transforms = Compose([LoadImageD(keys='eeg'),
                              SpatialCropD(keys='eeg',
                                           roi_slices=[slice(29952),
                                                       slice(29952)])])

    dataset = Dataset(data=file_list, transform=eeg_transforms)

    loader = DataLoader(
        dataset=dataset,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        shuffle=False,
    )
    return loader


def train(
        model,
        discriminator,
        optimizer_g,
        optimizer_d,
        dataloader,
        epoch,
        device,
        scaler_g,
        scaler_d,
        writer,
        config,
):
    model.train()
    discriminator.train()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, inputs in pbar:
        inputs = inputs["eeg"]
        optimizer_g.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            inputs = inputs.to(device)

            reconstruction, z_mu, z_sigma = model(x=inputs)
            loss_l1 = F.l1_loss(reconstruction.float(), inputs.float())

            loss_kl = 0.5 * torch.sum(
                z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
                dim=[1, 2],
            )
            loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]

            logits_fake = discriminator(reconstruction.contiguous().float())
            real_label = torch.ones_like(logits_fake, device=logits_fake.device)
            loss_g = F.mse_loss(logits_fake, real_label)

            loss = (
                    loss_l1
                    + config.model.kl_weight * loss_kl
                    + config.model.gan_weight * loss_g
            )

            loss = loss.mean()
            loss_l1 = loss_l1.mean()
            loss_kl = loss_kl.mean()
            loss_g = loss_g.mean()

            losses = OrderedDict(
                loss=loss,
                loss_l1=loss_l1,
                loss_kl=loss_kl,
                loss_g=loss_g,
            )

        scaler_g.scale(losses["loss"]).backward()
        scaler_g.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        scaler_g.step(optimizer_g)
        scaler_g.update()

        with autocast(enabled=True):
            logits_fake = discriminator(reconstruction.contiguous().detach())
            fake_label = torch.zeros_like(logits_fake, device=logits_fake.device)
            loss_d_fake = F.mse_loss(logits_fake, fake_label)
            logits_real = discriminator(inputs.contiguous().detach())
            real_label = torch.ones_like(logits_real, device=logits_real.device)
            loss_d_real = F.mse_loss(logits_real, real_label)

            loss_d = config.model.gan_weight * (loss_d_fake + loss_d_real) * 0.5

            loss_d = loss_d.mean()

        scaler_d.scale(loss_d).backward()
        scaler_d.unscale_(optimizer_d)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
        scaler_d.step(optimizer_d)
        scaler_d.update()

        writer.add_scalar("lr_g", get_lr(optimizer_g), epoch * len(dataloader) + step)
        writer.add_scalar("lr_d", get_lr(optimizer_d), epoch * len(dataloader) + step)
        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), epoch * len(dataloader) + step)

        # update pbar
        pbar.set_description(f"Epoch: {epoch}")
        pbar.set_postfix(
            loss=loss.item(),
            loss_g=loss_g.item(),
            loss_d_real=loss_d_real.item(),
        )


@torch.no_grad()
def validate(
        model,
        discriminator,
        dataloader,
        device,
        writer,
        config,
        step_register,
):
    model.eval()
    discriminator.eval()
    total_losses = OrderedDict()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, inputs in pbar:
        inputs = inputs["eeg"]

        inputs = inputs.to(device)

        with autocast(enabled=True):
            reconstruction, z_mu, z_sigma = model(x=inputs)
            loss_l1 = F.l1_loss(reconstruction.float(), inputs.float())

            loss_kl = 0.5 * torch.sum(
                z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
                dim=[1, 2],
            )
            loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]

            logits_fake = discriminator(reconstruction.contiguous().float())
            real_label = torch.ones_like(logits_fake, device=logits_fake.device)
            loss_g = F.mse_loss(logits_fake, real_label)

            logits_fake = discriminator(reconstruction.contiguous().detach())
            fake_label = torch.zeros_like(logits_fake, device=logits_fake.device)
            loss_d_fake = F.mse_loss(logits_fake, fake_label)
            logits_real = discriminator(inputs.contiguous().detach())
            real_label = torch.ones_like(logits_real, device=logits_real.device)
            loss_d_real = F.mse_loss(logits_real, real_label)
            loss_d = (loss_d_fake + loss_d_real) * 0.5

            loss = (
                    loss_l1
                    + config.model.kl_weight * loss_kl
                    + +config.model.gan_weight * loss_d
            )

            loss = loss.mean()
            loss_l1 = loss_l1.mean()
            loss_kl = loss_kl.mean()
            loss_g = loss_g.mean()
            loss_d = loss_d.mean()

            losses = OrderedDict(
                loss=loss,
                loss_l1=loss_l1,
                loss_kl=loss_kl,
                loss_g=loss_d,
                loss_d=loss_g,
            )

    for k, v in losses.items():
        total_losses[k] = total_losses.get(k, 0) + v.item() * inputs.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(dataloader.dataset)

    for k, v in total_losses.items():
        writer.add_scalar(f"{k}", v, step_register)

    log_reconstructions(
        img=inputs,
        recons=reconstruction,
        writer=writer,
        step=step_register,
    )

    return total_losses["loss_l1"]


def save_checkpoint(state, is_best, root):
    torch.save(state, root / "checkpoint.pth")
    if is_best:
        shutil.copyfile(root / "checkpoint.pth", "model_best.pth")


def main(args):
    config = OmegaConf.load(args.config_file)

    set_determinism(seed=config.train.seed)
    print_config()

    output_dir = Path("/project/outputs/runs/")
    output_dir.mkdir(exist_ok=True, parents=True)

    run_dir = output_dir / config.train.run_dir
    if run_dir.exists() and (run_dir / "checkpoint.pth").exists():
        resume = True
    else:
        resume = False
        run_dir.mkdir(exist_ok=True)

    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))

    data_path = Path(config.train.data_root)
    train_loader = train_dataloader(root=data_path, config=config)
    val_loader = test_dataloader(root=data_path, config=config)

    model = AutoencoderKL(**config["model"]["params"])
    discriminator = Discriminator(**config["discriminator"]["params"]).apply(
        weights_init
    )
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        discriminator = torch.nn.DataParallel(discriminator)

    model = model.to(device)
    discriminator = discriminator.to(device)

    # Optimizers
    optimizer_g = optim.Adam(model.parameters(), lr=config.model.base_lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config.model.disc_lr)

    # Get Checkpoint
    best_loss = float("inf")
    start_epoch = 0
    if resume:
        print("Using checkpoint!")
        checkpoint = torch.load(str(run_dir / "checkpoint.pth"))
        model.load_state_dict(checkpoint["state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
    else:
        print("No checkpoint found.")

    scaler_g = GradScaler()
    scaler_d = GradScaler()
    for epoch in range(start_epoch, config.train.n_epochs):
        print(f"Epoch {epoch}")
        train(
            model=model,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            dataloader=train_loader,
            epoch=epoch,
            device=device,
            scaler_g=scaler_g,
            scaler_d=scaler_d,
            writer=writer_train,
            config=config,
        )
        val_loss = validate(
            model=model,
            discriminator=discriminator,
            dataloader=val_loader,
            device=device,
            writer=writer_val,
            config=config,
            step_register=epoch * len(train_loader),
        )
        is_best = val_loss < best_loss

        if is_best:
            print(f"New best val loss {val_loss}")
            best_loss = val_loss
            raw_model = model.module if hasattr(model, "module") else model
            torch.save(raw_model.state_dict(), str(run_dir / "best_model.pth"))

        if (epoch + 1) % config.train.eval_freq == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "best_loss": best_loss,
                },
                is_best,
                run_dir,
            )

    log_mlflow(
        model=model,
        config=config,
        args=args,
        run_dir=run_dir,
        val_loss=val_loss,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
