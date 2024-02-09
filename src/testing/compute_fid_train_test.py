""" Script to compute the Frechet Inception Distance (FID) of the samples of the LDM.

In order to measure the quality of the samples, we use the Frechet Inception Distance (FID) metric between 1200 images
from the MIMIC-CXR dataset and 1000 images from the LDM.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from generative.metrics import FIDMetric
from monai import transforms
from monai.config import print_config
from monai.data import Dataset
from monai.utils import set_determinism
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
#from dataset.dataset import
from glob import glob
from braindecode.models import USleep
from dataset.dataset import test_dataloader
from numpy import load
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--sample_dir", help="Location of the samples to evaluate.")
    parser.add_argument("--test_ids", help="Location of file with test ids.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument(
        "--path_test_ids",
        type=str,
        default="/home/bru/PycharmProjects/DDPM-EEG/data/ids_shhs/ids_shhs_h_test.csv",
        #default="/project/data/ids/ids_sleep_edfx_cassette_train.csv",
    )
    parser.add_argument("--autoencoderkl_config_file_path",
                        help="Path to the .pth model from the stage1.",
                        #default="/project/config/config_aekl_eeg.yaml")
                        default="/home/bru/PycharmProjects/DDPM-EEG/config/config_aekl_eeg.yaml")
    parser.add_argument(
        "--path_pre_processed",
        type=str,
        default="/home/bru/PycharmProjects/DDPM-EEG/data/raw/shhs_h",
        #default="/data/pre-processed",
    )
    args = parser.parse_args()
    return args


def main(args):
    set_determinism(seed=args.seed)
    print_config()

    config_aekl = OmegaConf.load(args.autoencoderkl_config_file_path)

    # Load pretrained model
    device = torch.device("cpu")

    n_channels=2
    n_classes=5
    sfreq=100

    model = USleep(
        in_chans=n_channels,
        sfreq=sfreq,
        depth=12,
        with_skip_connection=True,
        n_classes=n_classes,
        input_size_s=30,
        apply_softmax=False
    )


    print(f"Defining the EEG Classifier.")

    params = torch.load("/home/bru/PycharmProjects/sleep-energy/runs/sleep_usleep/0/params.pt",
                        map_location=device)

    model.load_state_dict(params)


    # Samples
    test_loader = test_dataloader(config=config_aekl, args=args, upper_limit=1000, )

    test_loader_2 = test_dataloader(config=config_aekl, args=args, upper_limit=1000, )


    samples_features_1 = []
    samples_features_2 = []

    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for step, batch in pbar:
        x = batch['eeg'].to(device)
        for batch2 in test_loader_2:
            x2 = batch2["eeg"].to(device)
            if batch["eeg_meta_dict"]["filename_or_obj"][0] == batch2["eeg_meta_dict"]["filename_or_obj"][0]:
                continue

            x1_cropped = x[:, :, 36:-36]
            x2_cropped = x2[:, :, 36:-36]

            double_feature_space_1 = torch.concat([x1_cropped, x1_cropped], 1)
            double_feature_space_2 = torch.concat([x2_cropped, x2_cropped], 1)

            with torch.no_grad():
                _, outputs_1 = model(double_feature_space_1.to(device))
                _, outputs_2 = model(double_feature_space_2.to(device))

                outputs_1 = outputs_1.squeeze(-1)
                outputs_2 = outputs_2.squeeze(-1)

                # dense_outputs = F.adaptive_avg_pool1d(outputs.squeeze(-1).T,1).squeeze(-1)

            samples_features_1.append(outputs_1.cpu())
            samples_features_2.append(outputs_2.cpu())

        samples_features_1 = torch.cat(samples_features_1, dim=0)
        samples_features_2 = torch.cat(samples_features_2, dim=0)

        pbar.update()
    # Compute FID
    metric = FIDMetric()
    fid = metric(samples_features_1, samples_features_2)

    print(f"FID: {np.round(fid,6)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
