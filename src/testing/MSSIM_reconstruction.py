""" Script to compute the MS-SSIM score of the reconstructions of the Autoencoder.
"""

import argparse
from collections import OrderedDict
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from generative.networks.nets import AutoencoderKL
from monai.config import print_config
from monai.data import DataLoader, PersistentDataset
from monai.utils import set_determinism
from monai.transforms import Compose, LoadImageD, RandSpatialCropD, ScaleIntensityD, BorderPadD, EnsureChannelFirstD
from tqdm import tqdm
from omegaconf import OmegaConf
import pandas as pd
from pandas import DataFrame
from mne.filter import filter_data

from tailored_mssim import MultiScaleSSIMMetric

from MSSIM_test import test_dataloader
import mne
mne.set_log_level('CRITICAL')

sfreq = 100
windows_size = 30 * sfreq


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", help="Path where to find the config file.")
    parser.add_argument("--path_test_ids", help="Location of file with test ids.")
    parser.add_argument("--path_pre_processed", type=str, help="Path to preprocessed data")
    parser.add_argument("--best_model_path",
                        help="Path to the .pth model from the stage1.")
    parser.add_argument("--spe", type=str, default="no-spectral", choices=["spectral", "no-spectral"])
    parser.add_argument("--latent_channels", type=int, default=1, choices=[1,3])
    parser.add_argument("--dataset", help="Which dataset to use")
    args = parser.parse_args()
    return args


def main(args):
    print("Compute MSSIM on reconstruction...")
    config = OmegaConf.load(args.config_file)
    device = torch.device("cuda")

    print(args)
    set_determinism(seed=config.train.seed)
    print_config()

    print("Load the model")
    autoencoder_args = config.autoencoderkl.params
    autoencoder_args['num_channels'] = [32,32,64]
    autoencoder_args['latent_channels'] = args.latent_channels

    stage1 = AutoencoderKL(**autoencoder_args)

    state_dict = torch.load(args.best_model_path.format(args.spe, args.latent_channels)+"/best_model.pth",
                            map_location=device)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") # removing ‘.moldule’ from key
        new_state_dict[name] = v
    raw_model = stage1.module if hasattr(stage1, "module") else stage1
    raw_model.load_state_dict(new_state_dict)
    raw_model.to(device)

    print("Getting data...")
    test_loader = test_dataloader(config=config, args=args)

    device = torch.device("cuda")
    ms_ssim = MultiScaleSSIMMetric(spatial_dims=1, data_range=1.0, kernel_size=16)

    frequencies = ['delta', 'theta', 'alpha', 'all']


    print("Computing MS-SSIM...")
    frequency_data = {}
    for frequency in frequencies:
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        ms_ssim_list = []
        print(frequency)
        print("--------------------------------------------------")
        for step, batch in pbar:
            img = batch["eeg"].to(device)

            with torch.no_grad():
                img_recon = raw_model.reconstruct(img)
            
            # remove the 0-padding
            img_cropped = img[:,:,36:-36]
            img2_cropped = img_recon[:,:,36:-36]
       
            # Filter the data to include only one frequency
            if frequency == "delta":
                img_cropped = filter_data(img_cropped.detach().cpu().numpy().astype(float), sfreq=100, l_freq=0.5, h_freq=4.0)
                img2_cropped = filter_data(img2_cropped.detach().cpu().numpy().astype(float), sfreq=100, l_freq=0.5, h_freq=4.0)
            elif frequency == "theta":
                img_cropped = filter_data(img_cropped.detach().cpu().numpy().astype(float), sfreq=100, l_freq=4.1, h_freq=8.0)
                img2_cropped = filter_data(img2_cropped.detach().cpu().numpy().astype(float), sfreq=100, l_freq=4.1, h_freq=8.0)
            elif frequency == "alpha":
                img_cropped = filter_data(img_cropped.detach().cpu().numpy().astype(float), sfreq=100, l_freq=8.1, h_freq=12.0)
                img2_cropped = filter_data(img2_cropped.detach().cpu().numpy().astype(float), sfreq=100, l_freq=8.1, h_freq=12.0)
            elif frequency == "all":
                img_cropped = img_cropped.detach().cpu().numpy()
                img2_cropped = img2_cropped.detach().cpu().numpy()
            plt.plot(img_cropped[0, 0, :], label=frequency)
            frequency_data[frequency] = img_cropped[0, 0, :].astype(float)

            ms_ssim_list.append(ms_ssim(torch.from_numpy(img_cropped).to(device),
                                        torch.from_numpy(img2_cropped).to(device)).item())
            pbar.update()

        ms_ssim_list = np.array(ms_ssim_list)
        print(f"Mean MS-SSIM (reconstruction): {ms_ssim_list.mean():.3f}")
        print("")

    plt.savefig('/project/outputs/frequency_reconstruction.pdf')
    with open('/project/outputs/frequency_reconstruction.pickle', 'wb') as handle:
        pickle.dump(frequency_data, handle)

if __name__ == "__main__":
    args = parse_args()
    main(args)
