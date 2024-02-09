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
        default="/home/bru/PycharmProjects/DDPM-EEG/data/ids/ids_sleep_edfx_cassette_double_test.csv",
        #default="/project/data/ids/ids_sleep_edfx_cassette_train.csv",
    )
    parser.add_argument("--autoencoderkl_config_file_path",
                        help="Path to the .pth model from the stage1.",
                        #default="/project/config/config_aekl_eeg.yaml")
                        default="/home/bru/PycharmProjects/DDPM-EEG/config/config_aekl_eeg.yaml")
    parser.add_argument(
        "--path_pre_processed",
        type=str,
        default="/home/bru/PycharmProjects/DDPM-EEG/data/data_test",
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

    params = torch.load("/home/bru/PycharmProjects/sleep-energy/runs/sleep_usleep/0/final_model.pth",
                        map_location=device)

    model.load_state_dict(params)


    # Samples
    test_loader = test_dataloader(config=config_aekl, args=args)


    samples_features = []
    for batch in tqdm(test_loader):
        eeg_data = batch['eeg'].to(device)[:,:,36:-36]

        double_feature_space = torch.concat([eeg_data, eeg_data], 1)

        with torch.no_grad():
            predict, outputs = model(double_feature_space.to(device))
            outputs = outputs.squeeze(-1)
            #dense_outputs = F.adaptive_avg_pool1d(outputs.squeeze(-1).T,1).squeeze(-1)

        samples_features.append(outputs.cpu())
    samples_features = torch.cat(samples_features, dim=0)

    file_list = glob("/home/bru/PycharmProjects/DDPM-EEG/sample_synthetic/samples_ldm_1_spectral_sleep_edfx/sample*.npy")
    # 3 - spectral: 1.943942198930813
    # 1 - spectral: 1.2975230320143964
    # 3 - no spectral: 2.764283206833511
    # 1 - no spectral: 2.4693685146157804
    synthetic_features = []
    accumulating_data = []
    for file in file_list:
        accumulating_data.append(load(file))
        if len(accumulating_data) == 1000:
            continue

    merge_data = np.concatenate(accumulating_data[-64:], 0)
    eeg_data_sythetic = torch.from_numpy(merge_data)
    synthetic_features_double = torch.concat([eeg_data_sythetic, eeg_data_sythetic], 1)
    with torch.no_grad():
        predict, outputs = model(synthetic_features_double.to(device))
        outputs = outputs.squeeze(-1)
        #dense_outputs = F.adaptive_avg_pool1d(outputs.squeeze(-1).T,1).squeeze(-1)

    synthetic_features.append(outputs.cpu())
    synthetic_features = torch.cat(synthetic_features, dim=0)

    # Compute FID
    metric = FIDMetric()
    fid = metric(samples_features, synthetic_features)

    print(f"FID: {np.round(fid, 6)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
