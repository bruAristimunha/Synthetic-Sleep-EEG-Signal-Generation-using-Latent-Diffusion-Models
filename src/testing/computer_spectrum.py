"""
Author: Bruno Aristimunha
Training AutoEncoder KL with SleepEDFx or SHHS data.
Based on the tutorial from Monai Generative.

"""
import argparse

import pandas as pd
from monai.transforms import Compose, LoadImageD, RandSpatialCropSamplesD
from monai.utils import set_determinism
from omegaconf import OmegaConf

# print_config()

# for reproducibility purposes set a seed
set_determinism(42)

from monai.data import PersistentDataset
from pandas import DataFrame
from glob import glob

import mne
import numpy as np
import matplotlib.pyplot as plt
import torch
from monai.config import print_config
from monai.utils import set_determinism


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
        "--path_test_ids",
        type=str,
        #default="/home/bru/PycharmProjects/DDPM-EEG/data/ids/ids_sleep_edfx_cassette_valid.csv",
        default="/project/data/ids/ids_sleep_edfx_cassette_test.csv",
    )
    parser.add_argument(
        "--path_pre_processed",
        type=str,
        #default="/home/bru/PycharmProjects/DDPM-EEG/data/pre-processed",
        default="/data/pre-processed",
    )
    parser.add_argument(
        "--path_synthetic_data",
        type=str,
        #default="/home/bru/PycharmProjects/DDPM-EEG/output",
        default="/data/pre-processed",
    )

    parser.add_argument(
        "--path_output",
        type=str,
        #default="/home/bru/PycharmProjects/DDPM-EEG/analysis/",
        default="/data/pre-processed",
    )

    args = parser.parse_args()
    return args


def get_datalist(
        df: DataFrame, basepath: str
):
    """Get data dicts for data loaders.

    """

    data_dicts = []
    for index, row in df.iterrows():
        data_dicts.append(
            {
                "eeg": f"{basepath}/{row['FILE_NAME']}-PSG.npy",
                "subject": float(row["subject"]),
                "night": float(row["night"]),
                "age": float(row["age"]),
                "gender": float(row["gender"]),
                "lightoff": str(row["LightsOff"]),
            }
        )

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts


def test_dataloader(args):
    test_df = pd.read_csv(args.path_test_ids)

    test_dicts = get_datalist(test_df, basepath=args.path_pre_processed)
    sfreq = 100
    windows_size = 30 * sfreq

    transforms_list = Compose([LoadImageD(keys='eeg'),
                               RandSpatialCropSamplesD(keys='eeg',
                                                       num_samples=120,  # one hour
                                                       roi_size=[windows_size],
                                                       random_size=False,
                                                       ),
                               ])

    test_ds = PersistentDataset(data=test_dicts,
                                transform=transforms_list,
                                cache_dir=None)
    return test_ds


def main(args):
    config = OmegaConf.load(args.config_file)

    set_determinism(seed=config.train.seed)
    print_config()

    # Getting data loaders
    testdataloader = test_dataloader(args=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Getting the synthetic data
    file_list = glob(f"{args.path_synthetic_data}/*.npy")
    file_list.sort()
    print(f"Found {len(file_list)} synthetic files.")
    acc_synthetic = []
    for file in file_list:
        acc_synthetic.append(np.load(file))
    acc_synthetic = np.concatenate(acc_synthetic)

    info = mne.create_info(1, ch_types=['eeg'], sfreq=100)
    info['description'] = 'Original Dataset'
    synthetic_epochs = mne.EpochsArray(acc_synthetic, info)

    acc = []

    for test in testdataloader:
        acc.append(torch.cat([tes['eeg'] for tes in test]).to(device).numpy())

    acc = np.concatenate(acc)
    acc = acc.reshape((acc.shape[0], 1, acc.shape[1]))
    info = mne.create_info(1, ch_types=['eeg'], sfreq=100)
    info['description'] = 'Original Dataset'
    real_epochs = mne.EpochsArray(acc, info)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    spetrum_data_real = real_epochs.compute_psd()
    spetrum_data_synthetic = synthetic_epochs.compute_psd()

    spetrum_data_real.plot(axes=ax)
    spetrum_data_synthetic.plot(axes=ax)

    plt.savefig(f"{args.path_output}/psd_original_synthetic.pdf",
                bbox_inches='tight', pad_inches=0, dpi=300)


if __name__ == "__main__":
    args = parse_args()
    main(args)
