"""Utility functions for training."""
import argparse
import ast
from pathlib import Path, PosixPath

import mlflow.pytorch
import mne
from sklearn.metrics import roc_auc_score
from typing import Tuple, Union
from pynvml.smi import nvidia_smi
import joblib
import matplotlib.pyplot as plt
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from mlflow import start_run
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import autocast

class ParseListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parsed_list = ast.literal_eval(values)
        setattr(namespace, self.dest, parsed_list)


def setup_run_dir(config, args, base_path):
    # Create output directory

    output_dir = Path(base_path+config.train.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Extract number of channels from list from args
    # convert to string
    #num_channels = '-'.join(str(x) for x in args.num_channels)
    run_dir = output_dir / str(config.train.run_dir+"_"+args.spe+"_"+args.dataset)
    if run_dir.exists() and (run_dir / "checkpoint.pth").exists():
        resume = True
    else:
        resume = False
        run_dir.mkdir(exist_ok=True)

    return run_dir, resume


def recursive_items(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield from recursive_items(value)
        else:
            yield (key, value)


def print_gpu_memory_report():
    if torch.cuda.is_available():
        nvsmi = nvidia_smi.getInstance()
        data = nvsmi.DeviceQuery("memory.used, memory.total, utilization.gpu")["gpu"]
        print("Memory report")
        for i, data_by_rank in enumerate(data):
            mem_report = data_by_rank["fb_memory_usage"]
            print(f"gpu:{i} mem(%) {int(mem_report['used'] * 100.0 / mem_report['total'])}")


def file_convert_to_mne(numpy_epoch, name='Original Dataset', id_channel=None):
    """

    Parameters
    ----------
    list_epoch: list
    name: str

    Returns
    -------
    epochs: mne.EpochsArray
    """

    #numpy_epoch = np.concatenate(list_epoch)
    if id_channel is not None:
        ch_names = [f'EEG {id_channel}']
    else:
        ch_names = 1
    info = mne.create_info(ch_names, ch_types=['eeg'], sfreq=100)
    info['description'] = name

    epochs = mne.EpochsArray(numpy_epoch, info)

    return epochs


def get_epochs_spectrum(eeg_data, recons):
    import seaborn as sns
    sns.set_theme("poster")
    sns.set_style("white")
    
    eeg_data = eeg_data.cpu().numpy()
    recons = recons.cpu().numpy()

    epoch_original = file_convert_to_mne(eeg_data, name='Original Dataset')
    epoch_recont = file_convert_to_mne(recons, name='Reco Dataset')

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    legend_list = ['Origional', 'Reconstructed']
    colours = ['red', 'blue']
    spectral = epoch_original.compute_psd(fmax=12)
    spectral.plot(axes=ax, color="red", spatial_colors=False,
                  show=False, ci='range')

    spectral_rec = epoch_recont.compute_psd(fmax=12)
    spectral_rec.plot(axes=ax, color="blue", spatial_colors=False,
                  show=False, ci='range')

    plt.legend(ax.lines, legend_list, loc='upper right', labelcolor=colours)

    ax.set_yscale('log')

    ax.set_title('PSD of the original dataset and synthetic data')
    

    return fig, spectral, spectral_rec


def get_figure_ldm(sample, writer, step):
    sample = (
        sample.squeeze().cpu()
    )

    for i in range(sample.shape[0]):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        axes[0].plot(np.asarray(sample[i], dtype=np.float32))
        axes[0].set_title("Sample")
        #axes[0].axis("off")
        writer.add_figure(f"LDM_SAMPLE_{i}", fig, step)


def get_figure(
        img,
        recons,
):
    img = img[0, 0, :].cpu().numpy()
    recons = recons[0, 0, :].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    axes[0].plot(img)
    axes[0].set_title("Original")
    #axes[0].axis("off")
    axes[1].plot(recons)
    axes[1].set_title("Reconstruction")
    #axes[1].axis("off")

    return fig


def log_reconstructions(
        img: torch.Tensor,
        recons: torch.Tensor,
        writer: SummaryWriter,
        step: int,
        run_dir,
        name: str = "RECONSTRUCTION",
):
    fig = get_figure(
        img,
        recons,
    )
    writer.add_figure(f"{name}", fig, step)
    name_original = f"original_{name}_{step}.npy"
    name_reconstr = f"reconstr_{name}_{step}.npy"
    fig_name = f"original_{step}.pdf"
    np.save(str(run_dir / name_original), img)
    np.save(str(run_dir / name_reconstr), recons)

def log_spectral(
        eeg: torch.Tensor,
        recons: torch.Tensor,
        writer: SummaryWriter,
        run_dir,
        step: int,
        name: str = "SPECTRAL_RECONSTRUCTION",
):
    fig, spectral, spectral_rec  = get_epochs_spectrum(
        eeg,
        recons,
    )
    writer.add_figure(f"{name}", fig, step)
    name_original = f"original_spe_{name}_{step}.pkl"
    name_reconstr = f"reconstr_spe_{name}_{step}.pkl"
    fig_name = f"compare_{name}_{step}.pdf"
    fig.savefig(str(run_dir / fig_name), bbox_inches='tight')
    with open(str(run_dir / name_original), 'wb') as fo:  
        joblib.dump(spectral, fo)
    with open(str(run_dir / name_reconstr), 'wb') as fo:  
        joblib.dump(spectral_rec, fo)

def log_mlflow(
        model,
        config,
        args,
        run_dir: PosixPath,
        val_loss: float,
):
    experiment = config["train"]["experiment"]
    config = {**OmegaConf.to_container(config), **vars(args)}

    print(f"Setting mlflow experiment: {experiment}")
    mlflow.set_experiment(experiment)

    with start_run():
        print(f"MLFLOW URI: {mlflow.tracking.get_tracking_uri()}")
        print(f"MLFLOW ARTIFACT URI: {mlflow.get_artifact_uri()}")

        # for key, value in recursive_items(config):
        #     mlflow.log_param(key, value)

        mlflow.log_artifacts(str(run_dir), artifact_path="events")
        mlflow.log_metric(f"loss", val_loss, 0)

        raw_model = model.module if hasattr(model, "module") else model

        mlflow.pytorch.log_model(raw_model, "final_model")



@torch.no_grad()
def log_ldm_sample_unconditioned(
        model: nn.Module,
        stage1: nn.Module,
        scheduler: nn.Module,
        spatial_shape: Tuple,
        writer: SummaryWriter,
        step: int,
        device: torch.device,
        scale_factor: float = 1.0,
        images: torch.Tensor = None,
) -> None:
    latent = torch.randn((1,) + spatial_shape)
    latent = latent.to(device)

    for t in tqdm(scheduler.timesteps, ncols=70):
        noise_pred = model(x=latent, timesteps=torch.asarray((t,)).to(device))
        latent, _ = scheduler.step(noise_pred, t, latent)

    x_hat = stage1.model.decode(latent / scale_factor)
    x_hat_no_sacle = stage1.model.decode(latent)

    log_spectral(images, x_hat, writer, step+1, name="SAMPLE_UNCONDITIONED",)

    log_spectral(images, x_hat_no_sacle, writer, step+1, name="SAMPLE_NO_SCALE_UNCONDITIONED")

    log_spectral(x_hat, x_hat_no_sacle, writer, step+1, name="SAMPLE_COMPARE_SCALE_UNCONDITIONED")

    img_0 = x_hat[0, 0, :].cpu().numpy()
    fig = plt.figure(dpi=300)
    plt.plot(img_0)
    #plt.axis("off")
    writer.add_figure("SAMPLE", fig, step)


@torch.no_grad()
def log_diffusion_sample_unconditioned(
        model: nn.Module,
        scheduler: nn.Module,
        spatial_shape: Tuple,
        writer: SummaryWriter,
        step: int,
        device: torch.device,
        run_dir,
        inferer: object,
        images: torch.Tensor = None,
) -> None:
    latent = torch.randn((1,) + spatial_shape)
    latent = latent.to(device)
    scheduler.set_timesteps(num_inference_steps=1000)
    with autocast(enabled=True):
        images = inferer.sample(input_noise=latent, diffusion_model=model, scheduler=scheduler)

    log_spectral(eeg=images, recons=latent, writer=writer, step=step+1, name="SAMPLE_UNCONDITIONED", run_dir=run_dir)

    img_0 = latent[0, 0, :].cpu().numpy()
    fig = plt.figure(dpi=300)
    plt.plot(img_0)
    #plt.axis("off")
    writer.add_figure("SAMPLE", fig, step)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
