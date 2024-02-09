"""
Author: Bruno Aristimunha
Epochs Spectrum AutoEncoder KL with SleepEDFx or SHHS data.
Based on the tutorial from Monai Generative.
"""

import argparse
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf

from dataset.dataset import train_dataloader
from util import file_convert_to_mne

def parse_args():
    """
    Parsing the args from the command line.
    Returns
    -------

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default="/home/bru/PycharmProjects/DDPM-EEG/config/config_aekl_eeg.yaml",
        # default="/project/config/config_ldm.yaml",
        help="Path to config file with all the training parameters needed",
    )
    parser.add_argument(
        "--path_train_ids",
        type=str,
        default="/home/bru/PycharmProjects/DDPM-EEG/data/ids/ids_sleep_edfx_cassette_train.csv",
        # default="/project/data/ids/ids_sleep_edfx_cassette_test.csv",
    )

    parser.add_argument(
        "--path_synthetic_data",
        type=str,
        default="/home/bru/PycharmProjects/DDPM-EEG/outputs/samples/",
        # default="/data/pre-processed",
    )

    parser.add_argument(
        "--path_output",
        type=str,
        default="/home/bru/PycharmProjects/DDPM-EEG/analysis/",
        # default="/data/pre-processed",
    )

    parser.add_argument(
        "--path_pre_processed",
        type=str,
        default="/home/bru/PycharmProjects/DDPM-EEG/data/pre-processed",
        # default="/data/pre-processed",
    )
    parser.add_argument(
        "--path_cached_data",
        type=str,
        default="/home/bru/PycharmProjects/DDPM-EEG/data/pre",
        # default="/data/pre",
    )
    args = parser.parse_args()
    return args


def main(args):
    print_config()
    config = OmegaConf.load(args.config_file)

    set_determinism(seed=config.train.seed)
    # Getting data loaders
    testdataloader = train_dataloader(config=config, args=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Getting the synthetic data

    folder_list = glob(f"{args.path_synthetic_data}/*")
    folder_list.sort()
    print(f"Found {len(folder_list)} folders.")

    epochs_list = []
    for folder in folder_list:
        file_list = glob(f"{folder}/*.npy")

        list_epoch = [np.load(file) for file in file_list]

        epoch = file_convert_to_mne(list_epoch,
                                    name=folder.split('/')[-1])

        epochs_list.append((epoch, folder.split('/')[-1]))

    # Getting the real data
    real_data = []

    for test in testdataloader:
        data = [tes['eeg'] for tes in test]
        real_data.append(torch.cat(data).to(device).numpy())

    original_epoch = file_convert_to_mne(real_data,
                                         name='Original Dataset')

    epochs_list.append((original_epoch, 'Original Dataset'))

    print("Computing PSD")
    colours = ['blue', 'red', 'green', 'orange', 'purple']
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    legend_list = list()
    for (epoch, name), color in zip(epochs_list, colours):
        legend_list.append(name)
        spectral = epoch.compute_psd()
        spectral.plot(axes=ax, color=color, spatial_colors=False,
                      show=False, ci='range')

    plt.legend(ax.lines, legend_list, loc='upper right', labelcolor=colours)

    ax.set_yscale('log')
    ax.set_title('PSD of the original dataset and synthetic data')
    plt.show()

    fig.savefig(f"{args.path_output}/psd_original_synthetic.pdf",
                bbox_inches='tight', pad_inches=0, dpi=300)


if __name__ == "__main__":
    args = parse_args()
    main(args)
