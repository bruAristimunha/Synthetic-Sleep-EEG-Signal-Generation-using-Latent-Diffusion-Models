""" Training functions for the different models. """
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from pynvml.smi import nvidia_smi
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from util import log_diffusion_sample_unconditioned
from training.util import get_lr, print_gpu_memory_report
from generative.losses import JukeboxLoss
# ----------------------------------------------------------------------------------------------------------------------
# Latent Diffusion Model Unconditioned
# ----------------------------------------------------------------------------------------------------------------------
def train_diffusion_(
    model: nn.Module,
    scheduler: nn.Module,
    start_epoch: int,
    best_loss: float,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    eval_freq: int,
    writer_train: SummaryWriter,
    writer_val: SummaryWriter,
    device: torch.device,
    run_dir: Path,
    inferer: object,
    spectral_weight: float = None,
    spectral_loss: bool = False,
    scale_factor: float = 1.0,
) -> float:
    scaler = GradScaler()
    raw_model = model.module if hasattr(model, "module") else model

    val_loss = eval_diffusion(
        model=model,
        scheduler=scheduler,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        writer=writer_val,
        sample=False,
        scale_factor=scale_factor,
        inferer=inferer,
        spectral_weight=spectral_weight,
        spectral_loss=spectral_loss,
        run_dir=run_dir,
    )
    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")

    for epoch in range(start_epoch, n_epochs):
        train_epoch_diffusion(
            model=model,
            scheduler=scheduler,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer_train,
            scaler=scaler,
            scale_factor=scale_factor,
            inferer=inferer,
            spectral_weight=spectral_weight,
            spectral_loss=spectral_loss,
            run_dir=run_dir,
        )

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_diffusion(
                model=model,
                scheduler=scheduler,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val,
                sample=True if (epoch + 1) % (eval_freq * 2) == 0 else False,
                scale_factor=scale_factor,
                inferer=inferer,
                spectral_weight=spectral_weight,
                spectral_loss=spectral_loss,
                run_dir=run_dir,
            )

            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
            print_gpu_memory_report()

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "diffusion": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
                "scale_factor": scale_factor,
            }
            torch.save(checkpoint, str(run_dir / "checkpoint.pth"))

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss
                torch.save(raw_model.state_dict(), str(run_dir / "best_model.pth"))

    print(f"Training finished!")
    print(f"Saving final model...")
    torch.save(raw_model.state_dict(), str(run_dir / "final_model.pth"))

    return val_loss


def train_epoch_diffusion(
    model: nn.Module,
    scheduler: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    scaler: GradScaler,
    inferer,
    run_dir,
    spectral_weight,
    scale_factor: float = 1.0,
    spectral_loss: bool = False,
) -> None:
    model.train()

    pbar = tqdm(enumerate(loader), total=len(loader))
    jukebox_loss = JukeboxLoss(spatial_dims=1, reduction="sum")
    for step, x in pbar:

        images = x['eeg'].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            # Generate random noise
            noise = torch.randn_like(images).to(device)
            # Create timesteps
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()
            
            noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps) 
            
            if spectral_loss:
                recons_spectral = jukebox_loss(noise_pred.float(), noise.float())
                loss = F.mse_loss(noise_pred.float(), noise.float()) + recons_spectral*spectral_weight
            else:            
                loss = F.mse_loss(noise_pred.float(), noise.float())

        losses = OrderedDict(loss=loss)

        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()

        writer.add_scalar("lr", get_lr(optimizer), epoch * len(loader) + step)

        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

        pbar.set_postfix({"epoch": epoch, "loss": f"{losses['loss'].item():.5f}", "lr": f"{get_lr(optimizer):.6f}"})


@torch.no_grad()
def eval_diffusion(
    model: nn.Module,
    scheduler: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    writer: SummaryWriter,
    inferer,
    run_dir,
    spectral_weight,
    sample: bool = False,
    scale_factor: float = 1.0,
    spectral_loss: bool = False,
) -> float:
    model.eval()
    raw_model = model.module if hasattr(model, "module") else model
    total_losses = OrderedDict()
    jukebox_loss = JukeboxLoss(spatial_dims=1, reduction="sum")
    for x in loader:
        images = x["eeg"].to(device)	

        with autocast(enabled=True):

            # Generate random noise
            noise = torch.randn_like(images).to(device)
            # Create timesteps
            timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device).long()

            # Get model prediction
            noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
            
            if spectral_loss:
                recons_spectral = jukebox_loss(noise_pred.float(), noise.float())
                loss = F.mse_loss(noise_pred.float(), noise.float()) + recons_spectral*spectral_weight
            else:            
                loss = F.mse_loss(noise_pred.float(), noise.float())

        loss = loss.mean()
        losses = OrderedDict(loss=loss)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    for k, v in total_losses.items():
        writer.add_scalar(f"{k}", v, step)

    if sample:
        log_diffusion_sample_unconditioned(
            model=raw_model,
            scheduler=scheduler,
            spatial_shape=tuple(noise_pred.shape[1:]),
            writer=writer,
            step=step,
            device=device,
            images=images,
            inferer=inferer,
            run_dir=run_dir,
        )

    return total_losses["loss"]
