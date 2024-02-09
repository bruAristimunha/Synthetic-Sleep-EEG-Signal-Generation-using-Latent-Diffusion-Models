""" Script to compute the Frechet Inception Distance (FID) of the samples of the LDM.

In order to measure the quality of the samples, we use the Frechet Inception Distance (FID) metric between 1200 images
from the MIMIC-CXR dataset and 1000 images from the LDM.
"""
import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from generative.metrics import FIDMetric, MultiScaleSSIMMetric
from monai import transforms
from monai.config import print_config
from monai.data import Dataset
from monai.utils import set_determinism
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
#from dataset.dataset import
from braindecode.models import USleep

from dataset.dataset import test_dataloader
from generative.networks.nets import AutoencoderKL


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--output_dir", help="Location of the samples to evaluate.",
                        default="/project/metrics")
    parser.add_argument("--test_ids", help="Location of file with test ids.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument(
        "--path_test_ids",
        type=str,
        default="/project/data/ids_shhs/ids_shhs_test.csv",
        #default="/project/data/ids/ids_sleep_edfx_cassette_train.csv",
    )
    parser.add_argument("--autoencoderkl_config_file_path",
                        help="Path to the .pth model from the stage1.",
                        #default="/project/config/config_aekl_eeg.yaml")
                        default="/project/config/config_aekl_eeg.yaml")

    parser.add_argument("--best_model_path",
                        help="Path to the .pth model from the stage1.",
                        default="/project/outputs/aekl_eeg_{}_shhs_h_{}")

    parser.add_argument(
        "--path_pre_processed",
        type=str,
        #default="/home/bru/PycharmProjects/DDPM-EEG/data/raw/shhs_h",
        default="/data/polysomnography/shhs_numpy/",
    )

    parser.add_argument(
        "--spe",
        type=str,
        default="no-spectral",
        choices=["spectral", "no-spectral"]
    )
    parser.add_argument(
        "--latent_channels",
        type=int,
        default=1,
        choices=[1, 3]
    )
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device("cpu")

    set_determinism(seed=args.seed)
    print_config()

    config_path = Path(args.autoencoderkl_config_file_path)
    config_aekl = OmegaConf.load(str(config_path))
    autoencoder_args = config_aekl.autoencoderkl.params
    autoencoder_args['num_channels'] = [32,32,64]
    autoencoder_args['latent_channels'] = args.latent_channels

    stage1 = AutoencoderKL(**autoencoder_args)

    state_dict = torch.load(args.best_model_path.format(args.spe, args.latent_channels)+"/best_model.pth",
                            map_location=device)

    stage1.load_state_dict(state_dict)
    stage1.to(device)
    # Load pretrained model

    ms_ssim = MultiScaleSSIMMetric(spatial_dims=1, data_range=1.0, kernel_size=7)
    testdataloader = test_dataloader(config=config_aekl, args=args)

    print("Computing MS-SSIM...")
    ms_ssim_list = []
    filenames = []
    for batch in tqdm(testdataloader):
        x = batch['eeg'].to(device)

        with torch.no_grad():
            x_recon = stage1.reconstruct(x)

        x_cropped = x[:,:,36:-36]
        x_recon_cropped = x_recon[:,:,36:-36]

        ms_ssim_list.append(ms_ssim(x_cropped, x_recon_cropped))
        filenames.extend(batch['eeg_meta_dict']['filename_or_obj'])

    ms_ssim_list = torch.cat(ms_ssim_list, dim=0)

    prediction_df = pd.DataFrame({"filename": filenames, "ms_ssim": ms_ssim_list.cpu()[:, 0]})
    prediction_df.to_csv(Path(args.output_dir) / f"ms_ssim_reconstruction_shhs_{args.spe}_{args.latent_channels}.tsv",
                         index=False, sep="\t")

    print(f"Mean MS-SSIM {args.spe}_{args.latent_channels}: {ms_ssim_list.mean():.6f}")




if __name__ == "__main__":
    args = parse_args()
    main(args)
