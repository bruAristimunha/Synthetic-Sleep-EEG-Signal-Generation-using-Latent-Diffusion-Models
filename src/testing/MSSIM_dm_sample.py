""" Script to compute the MS-SSIM score of the synthetic images set."""

import argparse

import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from monai.transforms import Compose, LoadImageD, RandSpatialCropD, ScaleIntensityD, BorderPadD, EnsureChannelFirstD
from tqdm import tqdm
from omegaconf import OmegaConf
import pandas as pd
from pandas import DataFrame
from mne.filter import filter_data

from tailored_mssim import MultiScaleSSIMMetric

import mne
mne.set_log_level('CRITICAL')

transforms_list = Compose([LoadImageD(keys='eeg')])
def get_datalist(dataset, basepath):
    """
    Get data dicts for data loaders.

    """
    if dataset == "sleep_edfx":
        n = 64
    elif dataset == "shhs":
        n = 1000
        #n = 2288
    elif dataset == 'shhs_h':
        n = 132

    data_dicts = []
    for index in range(0, n):
        data_dicts.append(
            {
                "eeg": f"{basepath}/synthetic_trial_eeg_{index}.npy",
                "participant_id": index,
            }
        )

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts

def synthetic_dataloader(args, upper_limit=None):

    test_dicts = get_datalist(args.dataset, basepath=args.path_samples)

    test_ds = Dataset(data=test_dicts, transform=transforms_list)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                              num_workers=25, drop_last=False,
                              pin_memory=False,
                              persistent_workers=True)

    return test_loader

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path_samples", help="Location to the generated samples.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument("--dataset", help="Which dataset to use")

    args = parser.parse_args()
    return args


def main(args):
    print("Compute MSSIM on synthetic sample...")

    set_determinism(seed=2)
    print_config()

    print("Getting data...")
    test_loader = synthetic_dataloader(args=args)

    test_loader_2 = synthetic_dataloader(args=args)

    device = torch.device("cuda")
    ms_ssim = MultiScaleSSIMMetric(spatial_dims=1, data_range=1.0, kernel_size=16)

    print("Computing MS-SSIM...")
    frequencies = ['delta', 'theta', 'alpha', 'all']

    frequency_data = {}
    for frequency in frequencies:
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        ms_ssim_list = []
        print(frequency)
        print("--------------------------------------------------")
        for step, batch in pbar:
            img = batch["eeg"]
            # Filter the data to include only one frequency
            if frequency == "delta":
                img = filter_data(img[:, :, 0, :].numpy().astype(float), sfreq=100, l_freq=0.5, h_freq=4.0)
            elif frequency == "theta":
                img = filter_data(img[:, :, 0, :].numpy().astype(float), sfreq=100, l_freq=4.1, h_freq=8.0)
            elif frequency == "alpha":
                img = filter_data(img[:, :, 0, :].numpy().astype(float), sfreq=100, l_freq=8.1, h_freq=12.0)
            elif frequency == "all":
                img = img[:, :, 0, :].numpy()

            plt.plot(img[0, 0, :], label=frequency)
            frequency_data[frequency] = img[0, 0, :].astype(float)

            for batch2 in test_loader_2:
                img2 = batch2["eeg"]
                if batch["participant_id"] == batch2["participant_id"]:
                    continue
                # Filter the data to include only one frequency
                if frequency == "delta":
                    img2 = filter_data(img2[:, :, 0, :].astype(float), sfreq=100, l_freq=0.5, h_freq=4.0)
                elif frequency == "theta":
                    img2 = filter_data(img2[:, :, 0, :].numpy().astype(float), sfreq=100, l_freq=4.1, h_freq=8.0)
                elif frequency == "alpha":
                    img2 = filter_data(img2[:, :, 0, :].numpy().astype(float), sfreq=100, l_freq=8.1, h_freq=12.0)
                elif frequency == "all":
                    img2 = img2[:, :, 0, :].numpy()

                ms_ssim_list.append(ms_ssim(torch.from_numpy(img).to(device), 
                                            torch.from_numpy(img2).to(device)).item())
            pbar.update()

        ms_ssim_list = np.array(ms_ssim_list)
        print(f"Mean MS-SSIM: {ms_ssim_list.mean():.3f}")
        plt.savefig('/project/outputs/frequency_sample.pdf')
        with open('/project/outputs/frequency_sample.pickle', 'wb') as handle:
            pickle.dump(frequency_data, handle)

if __name__ == "__main__":
    args = parse_args()
    main(args)
