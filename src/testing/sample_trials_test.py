""" Script to generate sample EEG from the diffusion model.

"""

import argparse
from pathlib import Path

import torch
from generative.networks.nets import AutoencoderKL
from models.ldm import UNetModel
from generative.networks.schedulers import DDPMScheduler
from monai.config import print_config
from monai.utils import first, set_determinism
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.cpu.amp import autocast
import matplotlib.pyplot as plt
import numpy as np

from dataset.dataset import test_dataloader
from training.training import Stage1Wrapper
from util import file_convert_to_mne, ParseListAction


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", help="Path to save the .pth file of the diffusion model.",
                        # default = "/project/output")
                        default="/home/bru/PycharmProjects/DDPM-EEG/samples")
    parser.add_argument("--autoencoderkl_config_file_path",
                        help="Path to the .pth model from the stage1.",
                        #default="/project/config/config_aekl_eeg.yaml")
                        default="/home/bru/PycharmProjects/DDPM-EEG/config/config_aekl_eeg.yaml")

    parser.add_argument(
        "--path_test_ids",
        type=str,
        default="/home/bru/PycharmProjects/DDPM-EEG/data/ids/ids_sleep_edfx_cassette_double_test.csv",
        #default="/project/data/ids/ids_sleep_edfx_cassette_train.csv",
    )
    parser.add_argument(
        "--path_pre_processed",
        type=str,
        default="/home/bru/PycharmProjects/DDPM-EEG/data/data_test",
        #default="/data/pre-processed",
    )

    args = parser.parse_args()
    return args


def main(args):
    print_config()
    output_dir = Path(args.output_dir + "/samples_edfx_double")
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config_aekl = OmegaConf.load(args.autoencoderkl_config_file_path)

    test_loader = test_dataloader(config=config_aekl, args=args)
    channel_list = []
    for epoch in range(1, 100):
        for test_step, batch in enumerate(test_loader, start=1):
            eeg_data = batch['eeg'].to(device)

            print("Reading the EEG samples...")
            cropped_eeg_data = eeg_data.cpu().numpy()[:,:,36:-36]

            epoch_original = file_convert_to_mne(cropped_eeg_data, name='Reco Dataset')

            spectrum = epoch_original.compute_psd(fmax=18)

            #_, ax = plt.subplots()

            mean_spectrum = spectrum.average()
            psds, freqs = mean_spectrum.get_data(return_freqs=True)
            # then convert to dB and take mean & standard deviation across channels
            psds = 10 * np.log10(psds)
            psds_mean = psds.mean(axis=0)

            #psds_std = psds.std(axis=0)

            #ax.plot(freqs, psds_mean, color="k")
            save_info = [psds, freqs, psds_mean]
            channel_list.append(save_info)
            np.save(output_dir / f"psd_list_{epoch}.npy", save_info)


    print("Saved EEG trial.")

    np.save(output_dir / f"psd_list.npy", channel_list)


if __name__ == "__main__":
    args = parse_args()
    main(args)
