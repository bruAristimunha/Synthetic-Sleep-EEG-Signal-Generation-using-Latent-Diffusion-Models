""" Script to generate sample EEG from the diffusion model.

"""

import argparse
from pathlib import Path

import torch
from generative.networks.nets import AutoencoderKL
from models.ldm import UNetModel
from generative.networks.schedulers import DDIMScheduler
from monai.config import print_config
from monai.utils import first, set_determinism
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.cpu.amp import autocast
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from dataset.dataset import test_dataloader, train_dataloader
from training.training import Stage1Wrapper
from util import file_convert_to_mne, ParseListAction


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", help="Path to save the .pth file of the diffusion model.",
                        # default = "/project/output")
                        default="/project/sample_synthetic")

    parser.add_argument("--best_model_path",
                        help="Path to the .pth model from the stage1.",
                        default="/project/outputs/aekl_eeg_{}_{}_{}")
                        #aekl_eeg_no-spectral_shhs1_1
    parser.add_argument("--diffusion_path", help="Path to the .pth model from the diffusion model.",
                        #default="/project/trained_model/diffusion_model.pth")
                        default="/project/outputs/ldm_eeg_{}_{}_{}")
			 #ldm_eeg_no-spectral_shhs1_1
    parser.add_argument("--autoencoderkl_config_file_path",
                        help="Path to the .pth model from the stage1.",
                        #default="/project/config/config_aekl_eeg.yaml")
                        default="/project/config/config_aekl_eeg.yaml")

    parser.add_argument("--ldm_config_file_path",
                        help="Path to the .pth model from the LDM.",
                        #default="/project/config/config_ldm.yaml")
                        default="/project/config/config_ldm.yaml")


    parser.add_argument("--start_seed", type=int,
                        default=0)
    parser.add_argument("--stop_seed", type=int, default=1000)

    parser.add_argument("--guidance_scale", type=float, default=7.0, help="")

    parser.add_argument("--num_inference_steps", type=int, help="", default=200)

    parser.add_argument(
        "--path_pre_processed",
        type=str,
        #default="/home/bru/PycharmProjects/DDPM-EEG/data/pre-processed",
        default="/data/",
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
    parser.add_argument(
        "--type_dataset",
        type=str,
    )

    args = parser.parse_args()
    return args


def main(args):
    print_config()
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    output_dir = Path(args.output_dir + "/samples_ldm_{}_{}_{}".format(args.latent_channels, args.spe, args.type_dataset))
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config_aekl = OmegaConf.load(args.autoencoderkl_config_file_path.format(args.spe, args.type_dataset, args.latent_channels))
    autoencoder_args = config_aekl.autoencoderkl.params
    autoencoder_args['num_channels'] = [32,32,64]
    autoencoder_args['latent_channels'] = args.latent_channels

    stage1 = AutoencoderKL(**autoencoder_args)

    state_dict = torch.load(args.best_model_path.format(args.spe, args.type_dataset, args.latent_channels)+"/best_model.pth",
                            map_location=torch.device('cpu'))


    
    stage1.load_state_dict(state_dict)

    #stage1.load_state_dict(state_dict)
    stage1.to(device)


    config_ldm = OmegaConf.load(args.ldm_config_file_path)
    parameters = config_ldm['model']['params']['unet_config']['params']
    parameters['in_channels'] = args.latent_channels
    parameters['out_channels'] = args.latent_channels

    diffusion = UNetModel(**parameters)

    diffusion_path = args.diffusion_path.format(args.spe, args.type_dataset, args.latent_channels)

    state_dict_diffusion = torch.load(diffusion_path+"/best_model.pth",
                                      map_location=torch.device('cpu'))


    diffusion.load_state_dict(state_dict_diffusion)
    diffusion.to(device)
    diffusion.eval()

    checkpoint = torch.load(str(Path(diffusion_path) / "checkpoint.pth"),
                            map_location=torch.device('cpu'))
    scale_factor = checkpoint["scale_factor"]

    print(f"Scaling factor set to {scale_factor}")

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0205,
        schedule="scaled_linear_beta",
        prediction_type="v_prediction",
        clip_sample=False,
    )
    scheduler.set_timesteps(200)
    scheduler.to(device)
    #scale_factor = 1 / torch.std(z)

    channel_list = []
    for i in range(args.start_seed, args.stop_seed):
        set_determinism(seed=i)
        noise = torch.randn((1, args.latent_channels, 768)).to(device)

        with torch.no_grad():
            progress_bar = tqdm(scheduler.timesteps)
            for t in progress_bar:
                noise_input = noise
                model_output = diffusion(
                    noise_input, timesteps=torch.Tensor((t,)).to(noise.device).long()
                )
                #noise_pred_uncond, noise_pred_text = model_output.chunk(2)
                #noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

                noise, _ = scheduler.step(model_output, t, noise)

        with torch.no_grad():
            sample = stage1.decode_stage_2_outputs(noise / scale_factor)


        cropped_eeg_data = sample.cpu().numpy()[:, :, 36:-36]
        np.save(output_dir / f"sample_{i}.npy", cropped_eeg_data)

        epoch_original = file_convert_to_mne(cropped_eeg_data, name='Sythetic Dataset')

        spectrum = epoch_original.compute_psd(fmax=18)

        # _, ax = plt.subplots()

        mean_spectrum = spectrum.average()
        psds, freqs = mean_spectrum.get_data(return_freqs=True)
        # then convert to dB and take mean & standard deviation across channels
        psds = 10 * np.log10(psds)
        psds_mean = psds.mean(axis=0)

        #psds_std = psds.std(axis=0)

        plt.plot(freqs, psds_mean, color="k")
        plt.show()
        save_info = [psds, freqs, psds_mean]
        channel_list.append(save_info)
        np.save(output_dir / f"psd_list_{i}.npy", save_info)

        #np.save(output_dir / f"synthetic_trial_eeg_{i}.npy", synthetic_trial_eeg)
        print("Saved EEG trial.")

    print("Saved EEG trial.")

    np.save(output_dir / f"psd_list.npy", channel_list)
if __name__ == "__main__":
    args = parse_args()
    main(args)
