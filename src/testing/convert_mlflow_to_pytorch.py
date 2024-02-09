""" Script to convert the model from mlflow format to a format suitable for release (.pth).

All the following scripts will use the .pth format (easly shared).
"""
import argparse
from pathlib import Path

import mlflow.pytorch
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--stage1_mlflow_path",
                        help="Path to the MLFlow artifact of the stage1.",
                        #default="file:///home/bru/PycharmProjects/DDPM-EEG/mlruns/648037089844041024/585548f40673420e9c1749f35d0b7dda/artifacts/final_model",
                        default="file:///project/mlruns/648037089844041024/585548f40673420e9c1749f35d0b7dda/artifacts/final_model")

    parser.add_argument("--diffusion_mlflow_path", help="Path to the MLFlow artifact of the diffusion model.",
                        #default="file:///home/bru/PycharmProjects/DDPM-EEG/mlruns/607842016435220810/d54d33cc3aa2447785b0c24c2ff68083/artifacts/final_model",
                        default="file:///project/mlruns/607842016435220810/d54d33cc3aa2447785b0c24c2ff68083/artifacts/final_model")

    parser.add_argument("--output_dir", help="Path to save the .pth file of the diffusion model.",
                        default="/project/trained_model", )
                        #default="/home/bru/PycharmProjects/DDPM-EEG/trained_model")

    args = parser.parse_args()
    return args


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    stage1_model = mlflow.pytorch.load_model(args.stage1_mlflow_path, map_location=torch.device('cpu'))
    torch.save(stage1_model.state_dict(), output_dir / "autoencoder.pth")
    print("loaded autoencoder from mlflow")

    diffusion_model = mlflow.pytorch.load_model(args.diffusion_mlflow_path, map_location=torch.device('cpu'))
    torch.save(diffusion_model.state_dict(), output_dir / "diffusion_model.pth")
    print("loaded diffusion model from mlflow")


if __name__ == "__main__":
    args = parse_args()
    main(args)
