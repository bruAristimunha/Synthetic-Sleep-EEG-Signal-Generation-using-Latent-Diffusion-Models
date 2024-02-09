""" Script to compute the MS-SSIM score of the test set."""

import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
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

import mne
mne.set_log_level('CRITICAL')

sfreq = 100
windows_size = 30 * sfreq

def get_transforms(dataset):
    if dataset == 'sleep_edfx':
        transforms_list = Compose([LoadImageD(keys='eeg'),
                                   ScaleIntensityD(factor=1e6, keys='eeg'),  # Numeric stability
                                   ScaleIntensityD(minv=0, maxv=1, keys='eeg'),  # Normalization
                                   RandSpatialCropD(keys='eeg', roi_size=[windows_size],
                                                    random_size=False, ),
                                   BorderPadD(keys='eeg', spatial_border=[36], mode="constant")
                                   ])

    elif dataset in ['shhs_h', 'shhs']:
        transforms_list = Compose([LoadImageD(keys='eeg'),
                                   EnsureChannelFirstD(keys='eeg'),
                                   ScaleIntensityD(factor=1e6, keys='eeg'),  # Numeric stability
                                   ScaleIntensityD(minv=0, maxv=1, keys='eeg'),  # Normalization
                                   RandSpatialCropD(keys='eeg', roi_size=[windows_size],
                                                    random_size=False, ),
                                   BorderPadD(keys='eeg', spatial_border=[36], mode="constant")
                                   ])
    return transforms_list

def get_datalist(
        df: DataFrame, basepath: str
):
    """
    Get data dicts for data loaders.

    """
    data_dicts = []
    for index, row in df.iterrows():

        # check if there is an .npy ending on the file name name, if not add it
        eeg_file_name = f"{row['FILE_NAME_EEG']}" if row['FILE_NAME_EEG'].endswith('npy') else f"{row['FILE_NAME_EEG']}.npy" 
        data_dicts.append(
            {
                "eeg": f"{basepath}/{eeg_file_name}",
                "subject": float(row["subject"]),
                "night": float(row["night"]),
                "age": float(row["age"]),
                "gender": str(row["gender"]),
                "lightoff": str(row["LightsOff"]),
            }
        )

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts

def test_dataloader(config, args, upper_limit=None):
    test_df = pd.read_csv(args.path_test_ids)

    if upper_limit is not None:
        test_df = test_df[:upper_limit]

    test_dicts = get_datalist(test_df, basepath=args.path_pre_processed)
    
    transforms_list = get_transforms(args.dataset)
    test_ds = PersistentDataset(data=test_dicts, transform=transforms_list,
                                 cache_dir=None)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                              num_workers=config.train.num_workers, drop_last=config.train.drop_last,
                              pin_memory=False,
                              persistent_workers=True)

    return test_loader

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", help="Path where to find the config file.")
    parser.add_argument("--path_test_ids", help="Location of file with test ids.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument("--path_pre_processed", type=str, help="Path to preprocessed data")
    parser.add_argument("--dataset", help="Dataset to be used",
            choices=["sleep_edfx", "shhs_h", "shhs"])
    args = parser.parse_args()
    return args


def main(args):
    print("Running analysis of test samples...")
    config = OmegaConf.load(args.config_file)

    set_determinism(seed=config.train.seed)
    print_config()

    print("Getting data...")
    test_loader = test_dataloader(config=config, args=args)
    test_loader_2 = test_dataloader(config=config, args=args)

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
            img = batch["eeg"]
            for batch2 in test_loader_2:
                img2 = batch2["eeg"]
                if batch["eeg_meta_dict"]["filename_or_obj"][0] == batch2["eeg_meta_dict"]["filename_or_obj"][0]:
                    continue
                # remove the 0-padding
                img_cropped = img[:,:,36:-36]
                img2_cropped = img2[:,:,36:-36]
               
                # Filter the data to include only one frequency
                if frequency == "delta":
                    img_cropped = filter_data(img_cropped.numpy().astype(float), sfreq=100, l_freq=0.5, h_freq=4.0)
                    img2_cropped = filter_data(img2_cropped.numpy().astype(float), sfreq=100, l_freq=0.5, h_freq=4.0)
                elif frequency == "theta":
                    img_cropped = filter_data(img_cropped.numpy().astype(float), sfreq=100, l_freq=4.1, h_freq=8.0)
                    img2_cropped = filter_data(img2_cropped.numpy().astype(float), sfreq=100, l_freq=4.1, h_freq=8.0)
                elif frequency == "alpha":
                    img_cropped = filter_data(img_cropped.numpy().astype(float), sfreq=100, l_freq=8.1, h_freq=12.0)
                    img2_cropped = filter_data(img2_cropped.numpy().astype(float), sfreq=100, l_freq=8.1, h_freq=12.0)
                elif frequency == "all":
                    img_cropped = img_cropped.numpy()
                    img2_cropped = img2_cropped.numpy()
                plt.plot(img_cropped[0, 0, :], label=frequency)
                frequency_data[frequency] = img_cropped[0, 0, :].astype(float)

                ms_ssim_list.append(ms_ssim(torch.from_numpy(img_cropped).to(device), 
                                            torch.from_numpy(img2_cropped).to(device)).item())
            pbar.update()

        ms_ssim_list = np.array(ms_ssim_list)
        print(f"Mean MS-SSIM: {ms_ssim_list.mean():.3f}")
        print("")
    
    plt.savefig(f'/project/outputs/{args.dataset}_frequency_test.pdf')
    with open(f'/project/outputs/{args.dataset}_frequency_test.pickle', 'wb') as handle:
        pickle.dump(frequency_data, handle)

if __name__ == "__main__":
    args = parse_args()
    main(args)
