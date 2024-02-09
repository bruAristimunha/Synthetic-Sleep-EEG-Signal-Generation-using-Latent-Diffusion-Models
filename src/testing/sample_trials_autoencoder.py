""" Script to generate sample EEG from the diffusion model.

"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from generative.networks.nets import AutoencoderKL
from monai.config import print_config
from omegaconf import OmegaConf
from collections import OrderedDict

from dataset.dataset import train_dataloader
from util import ParseListAction


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", help="Path to save the .pth file of the diffusion model.",
                        default="/home/bru/PycharmProjects/DDPM-EEG/outputs/")
                        # default="/project/output")

    parser.add_argument("--stage1_path", help="Path to the .pth model from the stage1.",
                        # default="/project/trained_model/autoencoder.pth")
                        default="/home/bru/PycharmProjects/DDPM-EEG/aekl_bigger/aekl_eeg_{}/best_model.pth")

    parser.add_argument("--autoencoderkl_config_file_path",
                        help="Path to the .pth model from the stage1.",
                        # default="/project/config/config_aekl_eeg.yaml")
                        default="/home/bru/PycharmProjects/DDPM-EEG/config/config_aekl_eeg.yaml")

    parser.add_argument(
        "--num_channels",
        type=str, action=ParseListAction
    )
    parser.add_argument(
        "--path_train_ids",
        type=str,
        default="/home/bru/PycharmProjects/DDPM-EEG/data/ids/ids_sleep_edfx_cassette_train.csv",
        # default="/project/data/ids/ids_sleep_edfx_cassette_test.csv",
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

    output_dir = Path(args.output_dir + "/samples")
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = OmegaConf.load(args.autoencoderkl_config_file_path)
    autoencoder_args = config.autoencoderkl.params
    autoencoder_args['num_channels'] = args.num_channels

    stage1 = AutoencoderKL(**autoencoder_args)
    testdataloader = train_dataloader(config=config, args=args)

    models_channels = '-'.join(str(item)
                               for item in autoencoder_args['num_channels'])

    parameters_models = args.stage1_path.format(models_channels)

    state_dict = torch.load(parameters_models,
                            map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    stage1.load_state_dict(new_state_dict)
    stage1.to(device)
    stage1.eval()

    for i, test in enumerate(testdataloader):
        eeg_data = torch.cat([x['eeg'] for x in test]).to(device)
        with torch.no_grad():
            recontr, _, _ = stage1(eeg_data)

        synthetic_trial_eeg = recontr.numpy()
        img = eeg_data.numpy()[0][0]
        rec = recontr.numpy()[0][0]

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].plot(img)
        axes[0].set_title("Original")
        axes[1].plot(rec)
        axes[1].set_title("Reconstruction")
        plt.show()
        sample_folder = output_dir / f"{models_channels}"

        if not sample_folder.exists():
            sample_folder.mkdir(parents=True)

        np.save(sample_folder / f"synthetic_trial_eeg_{i}.npy", synthetic_trial_eeg)
        print("Saved EEG trial.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
