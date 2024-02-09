""" Script to generate sample EEG from the diffusion model.

"""

import argparse
from pathlib import Path
import os
import torch
from generative.inferers import DiffusionInferer
from generative.networks.schedulers import DDIMScheduler
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np
from models.ldm import UNetModel
from util import file_convert_to_mne

if os.path.exists('/project'):
    base_path = '/project/'
    base_path_data = '/data/'
else:
    base_path = '/home/bru/PycharmProjects/DDPM-EEG/'
    base_path_data = base_path
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir",
                        help="Path to save the .pth file of the diffusion model.",
                        default=f"{base_path}/sample_synthetic")

    parser.add_argument("--config_file",
                        help="Path to the .pth model from the DM.",
                        default=f"{base_path}/config/config_dm.yaml")

    parser.add_argument("--start_seed", type=int,
                        default=0)
    parser.add_argument("--stop_seed", type=int,
                        default=1000)
    parser.add_argument("--num_inference_steps",
                        type=int, help="", default=1000)

    parser.add_argument(
        "--spe",
        type=str,
        default="no-spectral",
        choices=["spectral", "no-spectral"]
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="edfx",
        choices=["edfx", "shhs", "shhsh"]
    )
    args = parser.parse_args()
    return args


def main(args):
    print_config()
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    output_dir = Path(args.output_dir + f"/samples_dm_{args.spe}_{args.dataset}")
    output_dir.mkdir(exist_ok=True, parents=True)
    print(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = OmegaConf.load(args.config_file)

    parameters = config['model']['params']['unet_config']['params']
    parameters['in_channels'] = 1
    parameters['out_channels'] = 1

    diffusion = UNetModel(**parameters)

    diffusion_path = Path(f"{base_path}/project/outputs_dm/dm_eeg_{args.spe}_{args.dataset}/best_model.pth")
    state_dict_diffusion = torch.load(diffusion_path,
                                      map_location=torch.device('cpu'))
    diffusion.load_state_dict(state_dict_diffusion)
    diffusion.to(device)
    diffusion.eval()

    scheduler = DDIMScheduler(
        num_train_timesteps=args.num_inference_steps,
        beta_start=0.0015,
        beta_end=0.0205,
        schedule="scaled_linear_beta",
        prediction_type="v_prediction",
        clip_sample=False,
    )
    scheduler.set_timesteps(200)
    scheduler.to(device)
    inferer = DiffusionInferer(scheduler)
    diffusion.eval()

    channel_list = []
    for i in range(args.start_seed, args.stop_seed):
        set_determinism(seed=i)
        noise = torch.randn((1, 1, 3072)).to(device)

        sample = inferer.sample(input_noise=noise, diffusion_model=diffusion,
                                scheduler=scheduler)

        cropped_eeg_data = sample.cpu().numpy()[:, :, 36:-36]
        np.save(output_dir / f"sample_{i}.npy", cropped_eeg_data)

        epoch_original = file_convert_to_mne(cropped_eeg_data,
                                             name='Sythetic Dataset')

        spectrum = epoch_original.compute_psd(fmax=18)

        # _, ax = plt.subplots()

        mean_spectrum = spectrum.average()
        psds, freqs = mean_spectrum.get_data(return_freqs=True)
        # then convert to dB and take mean & standard deviation across channels
        psds = 10 * np.log10(psds)
        psds_mean = psds.mean(axis=0)

        plt.plot(freqs, psds_mean, color="k")
        plt.show()
        save_info = [psds, freqs, psds_mean]
        channel_list.append(save_info)
        np.save(output_dir / f"psd_list_{i}.npy", save_info)
        np.save(output_dir / f"synthetic_trial_eeg_{i}.npy", cropped_eeg_data)
    print("Saved EEG trial.")

    np.save(output_dir / f"psd_list.npy", channel_list)

if __name__ == "__main__":
    args = parse_args()
    main(args)
