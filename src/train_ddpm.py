from collections import OrderedDict
from pathlib import Path

import mlflow.pytorch
import torch
import torch.optim as optim
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from models.ldm import DDPM
from train_encoder import get_lr, parse_args, test_dataloader, train_dataloader
from util import log_ldm_sample, log_mlflow


def train(
    model,
    eeg_encoder,
    loader,
    optimizer,
    epoch,
    writer,
    device,
    scaler,
):
    model.train()
    eeg_encoder.eval()

    pbar = tqdm(enumerate(loader), total=len(loader))

    for step, inputs in pbar:
        image = inputs["eeg"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            with torch.no_grad():
                z_image = eeg_encoder(x=image, get_ldm_inputs=True)
                z = z_image
            loss = model(z)[0].mean()
        losses = OrderedDict(loss=loss)

        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()

        writer.add_scalar("lr", get_lr(optimizer), epoch * len(loader) + step)
        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

        pbar.set_postfix(
            {
                "epoch": epoch,
                "loss": f"{losses['loss'].item():.5f}",
                "lr": f"{get_lr(optimizer):.6f}",
            }
        )
    return losses["loss"]


def validate(
    model,
    eeg_encoder,
    loader,
    epoch,
    writer,
    device,
    config,
    epoch_step,
):
    model.eval()
    raw_eeg_encoder = (
        eeg_encoder.module if hasattr(eeg_encoder, "module") else eeg_encoder
    )

    total_losses = OrderedDict()

    pbar = tqdm(enumerate(loader), total=len(loader))

    for step, inputs in pbar:
        image = inputs["eeg"].to(device)
        with autocast(enabled=True):
            z_image = eeg_encoder(x=image, get_ldm_inputs=True)
            z = z_image
            loss = model(z)[0].mean()

        losses = OrderedDict(loss=loss)

        for k, v in losses.items():
            total_losses[k] = (
                total_losses.get(k, 0) + v.item() * image.shape[0]
            )

        pbar.set_postfix(
            {"epoch": epoch, "loss": f"{losses['loss'].item():.5f}"}
        )

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    for k, v in total_losses.items():
        writer.add_scalar(f"{k}", v, epoch_step)

    # if epoch % config.train.save_every == 0:
    raw_model = model.module if hasattr(model, "module") else model

    log_ldm_sample(
        diffusion_model=raw_model,
        stage1_model_img=raw_eeg_encoder,
        spatial_shape=list(z.shape[1:]),
        writer=writer,
        step=epoch_step
    )

    return total_losses["loss"]


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

    model = DDPM(**config["model"]["params"])
    eeg_encoder = mlflow.pytorch.load_model(config.train.path_image,  map_location='cpu')
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        eeg_encoder = torch.nn.DataParallel(eeg_encoder)

    model = model.to(device)
    eeg_encoder = eeg_encoder.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.train.base_lr)

    # Get Checkpoint
    best_loss = float("inf")
    val_loss = float("inf")
    start_epoch = 0
    if resume:
        print("Using checkpoint!")
        checkpoint = torch.load(str(run_dir / "checkpoint.pth"))
        model.load_state_dict(checkpoint["diffusion"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
    else:
        print("No checkpoint found.")

    scaler = GradScaler()
    for epoch in range(start_epoch, config.train.n_epochs):
        print(f"Epoch {epoch}")

        train_loss = train(
            model,
            eeg_encoder,
            train_loader,
            optimizer,
            epoch,
            writer_train,
            device,
            scaler,
        )

        print(train_loss)

        if (epoch + 1) % config.train.eval_freq == 0:
            val_loss = validate(
                model,
                eeg_encoder,
                val_loader,
                epoch,
                writer_val,
                device,
                config,
                epoch * len(train_loader),
            )

        if (epoch + 1) % config.train.save_every == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "diffusion": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            }
            torch.save(checkpoint, str(run_dir / "checkpoint.pth"))
            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss
                raw_model = model.module if hasattr(model, "module") else model
                torch.save(
                    raw_model.state_dict(), str(run_dir / "best_model.pth")
                )

    print("Training finished!")
    print("Saving final model...")
    torch.save(model.state_dict(), str(run_dir / "final_model.pth"))
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
