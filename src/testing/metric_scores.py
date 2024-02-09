import torch
from generative.metrics import MultiScaleSSIMMetric


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--sample_dir", help="Location of the samples to evaluate.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")

    args = parser.parse_args()
    return args

def main(args):
    # Settings
    set_determinism(args.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    print(f"Using Device")

    # Load the models
    #-------------------------------------------------------------------------------
    # autoencoderKL

    # LDM

    # Load data
    #-------------------------------------------------------------------------------

    # Compute MS-SSIM between the reconstructed images and the original images
    #-------------------------------------------------------------------------------
    ms_ssim_recon_scores = []

    ms_ssim = MultiScaleSSIMMetric(spatial_dimension=1, data_range=1.0, kernel_size=4)
    for step, x in list(enumerate(val_loader)):
        image = x["image"].to(device)

        with torch.no_grad():
            image_recon = autoencoderkl.reconstruct(image)

        ms_ssim_recon_scores.append(ms_ssim(image, image_recon))

    ms_ssim_recon_scores = torch.cat(ms_ssim_recon_scores, dim=0)
    print(f"MS-SSIM Metric: {ms_ssim_recon_scores.mean():.4f} +- {ms_ssim_recon_scores.std():.4f}")

    # Compute MS-SSIM betwen pairs of synthetic images
    #-------------------------------------------------------------------------------
    ms_ssim_recon_scores = []

    # Load the synthetic images (or generate some)

    # iterate over the pair of images

    ms_ssim_recon_scores = torch.cat(ms_ssim_recon_scores, dim=0)
    print(f"MS-SSIM Metric: {ms_ssim_recon_scores.mean():.4f} +- {ms_ssim_recon_scores.std():.4f}")



if __name__ == "__main__":
    args = parse_args()
    main(args)

