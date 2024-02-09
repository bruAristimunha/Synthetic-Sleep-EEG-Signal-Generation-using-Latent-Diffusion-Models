from __future__ import annotations
""" Script to compute the Frechet Inception Distance (FID) of the samples of the LDM.

In order to measure the quality of the samples, we use the Frechet Inception Distance (FID) metric between 1200 images
from the MIMIC-CXR dataset and 1000 images from the LDM.
"""
import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F

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
from collections import OrderedDict
import numpy as np

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections.abc import Sequence

import torch
import torch.nn.functional as F
from monai.metrics.regression import RegressionMetric
from monai.utils import MetricReduction, StrEnum, ensure_tuple_rep

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections.abc import Sequence

import torch
import torch.nn.functional as F
from monai.metrics.regression import RegressionMetric
from monai.utils import MetricReduction, StrEnum, convert_data_type, ensure_tuple_rep
from monai.utils.type_conversion import convert_to_dst_type


class KernelType(StrEnum):
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"


class SSIMMetric(RegressionMetric):
    r"""
    Computes the Structural Similarity Index Measure (SSIM).

    .. math::
        \operatorname {SSIM}(x,y) =\frac {(2 \mu_x \mu_y + c_1)(2 \sigma_{xy} + c_2)}{((\mu_x^2 + \
                \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}

    For more info, visit
        https://vicuesoft.com/glossary/term/ssim-ms-ssim/

    SSIM reference paper:
        Wang, Zhou, et al. "Image quality assessment: from error visibility to structural
        similarity." IEEE transactions on image processing 13.4 (2004): 600-612.

    Args:
        spatial_dims: number of spatial dimensions of the input images.
        data_range: value range of input images. (usually 1.0 or 255)
        kernel_type: type of kernel, can be "gaussian" or "uniform".
        kernel_size: size of kernel
        kernel_sigma: standard deviation for Gaussian kernel.
        k1: stability constant used in the luminance denominator
        k2: stability constant used in the contrast denominator
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans)
    """

    def __init__(
        self,
        spatial_dims: int,
        data_range: float = 1.0,
        kernel_type: KernelType | str = KernelType.GAUSSIAN,
        kernel_size: int | Sequence[int, ...] = 11,
        kernel_sigma: float | Sequence[float, ...] = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)

        self.spatial_dims = spatial_dims
        self.data_range = data_range
        self.kernel_type = kernel_type

        if not isinstance(kernel_size, Sequence):
            kernel_size = ensure_tuple_rep(kernel_size, spatial_dims)
        self.kernel_size = kernel_size

        if not isinstance(kernel_sigma, Sequence):
            kernel_sigma = ensure_tuple_rep(kernel_sigma, spatial_dims)
        self.kernel_sigma = kernel_sigma

        self.k1 = k1
        self.k2 = k2

    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted image.
                It must be a 2D or 3D batch-first tensor [B,C,H,W] or [B,C,H,W,D].
            y: Reference image.
                It must be a 2D or 3D batch-first tensor [B,C,H,W] or [B,C,H,W,D].

        Raises:
            ValueError: when `y_pred` is not a 2D or 3D image.
        """
        dims = y_pred.ndimension()
        if self.spatial_dims == 2 and dims != 4:
            raise ValueError(
                f"y_pred should have 4 dimensions (batch, channel, height, width) when using {self.spatial_dims} "
                f"spatial dimensions, got {dims}."
            )

        if self.spatial_dims == 3 and dims != 5:
            raise ValueError(
                f"y_pred should have 4 dimensions (batch, channel, height, width, depth) when using {self.spatial_dims}"
                f" spatial dimensions, got {dims}."
            )

        ssim_value_full_image, _ = compute_ssim_and_cs(
            y_pred=y_pred,
            y=y,
            spatial_dims=self.spatial_dims,
            data_range=self.data_range,
            kernel_type=self.kernel_type,
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            k1=self.k1,
            k2=self.k2,
        )

        ssim_per_batch: torch.Tensor = ssim_value_full_image.view(ssim_value_full_image.shape[0], -1).mean(
            1, keepdim=True
        )

        return ssim_per_batch


def _gaussian_kernel(
    spatial_dims: int, num_channels: int, kernel_size: Sequence[int, ...], kernel_sigma: Sequence[float, ...]
) -> torch.Tensor:
    """Computes 2D or 3D gaussian kernel.

    Args:
        spatial_dims: number of spatial dimensions of the input images.
        num_channels: number of channels in the image
        kernel_size: size of kernel
        kernel_sigma: standard deviation for Gaussian kernel.
    """

    def gaussian_1d(kernel_size: int, sigma: float) -> torch.Tensor:
        """Computes 1D gaussian kernel.

        Args:
            kernel_size: size of the gaussian kernel
            sigma: Standard deviation of the gaussian kernel
        """
        dist = torch.arange(start=(1 - kernel_size) / 2, end=(1 + kernel_size) / 2, step=1)
        gauss = torch.exp(-torch.pow(dist / sigma, 2) / 2)
        return (gauss / gauss.sum()).unsqueeze(dim=0)

    gaussian_kernel_x = gaussian_1d(kernel_size[0], kernel_sigma[0])
    print("IMPORTAR")
    #gaussian_kernel_y = gaussian_1d(kernel_size[1], kernel_sigma[1])
    kernel = gaussian_kernel_x#torch.matmul(gaussian_kernel_x.t(), gaussian_kernel_y)  # (kernel_size, 1) * (1, kernel_size)

    kernel_dimensions = (num_channels, 1, kernel_size[0])# kernel_size[1])

    if spatial_dims == 3:
        gaussian_kernel_z = gaussian_1d(kernel_size[2], kernel_sigma[2])[None,]
        kernel = torch.mul(
            kernel.unsqueeze(-1).repeat(1, 1, kernel_size[2]),
            gaussian_kernel_z.expand(kernel_size[0], kernel_size[1], kernel_size[2]),
        )
        kernel_dimensions = (num_channels, 1, kernel_size[0], kernel_size[1], kernel_size[2])

    return kernel.expand(kernel_dimensions)


def compute_ssim_and_cs(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    spatial_dims: int,
    data_range: float = 1.0,
    kernel_type: KernelType | str = KernelType.GAUSSIAN,
    kernel_size: Sequence[int, ...] = 11,
    kernel_sigma: Sequence[float, ...] = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Function to compute the Structural Similarity Index Measure (SSIM) and Contrast Sensitivity (CS) for a batch
    of images.

    Args:
        y_pred: batch of predicted images with shape (batch_size, channels, spatial_dim1, spatial_dim2[, spatial_dim3])
        y: batch of target images with shape (batch_size, channels, spatial_dim1, spatial_dim2[, spatial_dim3])
        spatial_dims: number of spatial dimensions of the images (2, 3)
        data_range: the data range of the images.
        kernel_type: the type of kernel to use for the SSIM computation. Can be either "gaussian" or "uniform".
        kernel_size: the size of the kernel to use for the SSIM computation.
        kernel_sigma: the standard deviation of the kernel to use for the SSIM computation.
        k1: the first stability constant.
        k2: the second stability constant.

    Returns:
        ssim: the Structural Similarity Index Measure score for the batch of images.
        cs: the Contrast Sensitivity for the batch of images.
    """
    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    y_pred = convert_data_type(y_pred, output_type=torch.Tensor, dtype=torch.float)[0]
    y = convert_data_type(y, output_type=torch.Tensor, dtype=torch.float)[0]

    num_channels = y_pred.size(1)

    if kernel_type == KernelType.GAUSSIAN:
        kernel = _gaussian_kernel(spatial_dims, num_channels, kernel_size, kernel_sigma)
    elif kernel_type == KernelType.UNIFORM:
        kernel = torch.ones((num_channels, 1, *kernel_size)) / torch.prod(torch.tensor(kernel_size))

    kernel = convert_to_dst_type(src=kernel, dst=y_pred)[0]

    c1 = (k1 * data_range) ** 2  # stability constant for luminance
    c2 = (k2 * data_range) ** 2  # stability constant for contrast

    conv_fn = getattr(F, f"conv{spatial_dims}d")
    mu_x = conv_fn(y_pred, kernel, groups=num_channels)
    mu_y = conv_fn(y, kernel, groups=num_channels)
    mu_xx = conv_fn(y_pred * y_pred, kernel, groups=num_channels)
    mu_yy = conv_fn(y * y, kernel, groups=num_channels)
    mu_xy = conv_fn(y_pred * y, kernel, groups=num_channels)

    sigma_x = mu_xx - mu_x * mu_x
    sigma_y = mu_yy - mu_y * mu_y
    sigma_xy = mu_xy - mu_x * mu_y

    contrast_sensitivity = (2 * sigma_xy + c2) / (sigma_x + sigma_y + c2)
    ssim_value_full_image = ((2 * mu_x * mu_y + c1) / (mu_x**2 + mu_y**2 + c1)) * contrast_sensitivity

    return ssim_value_full_image, contrast_sensitivity
    

class KernelType(StrEnum):
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"


class MultiScaleSSIMMetric(RegressionMetric):
    """
    Computes the Multi-Scale Structural Similarity Index Measure (MS-SSIM).

    [1] Wang, Z., Simoncelli, E.P. and Bovik, A.C., 2003, November.
            Multiscale structural similarity for image quality assessment.
            In The Thirty-Seventh Asilomar Conference on Signals, Systems
            & Computers, 2003 (Vol. 2, pp. 1398-1402). Ieee.

    Args:
        spatial_dims: number of spatial dimensions of the input images.
        data_range: value range of input images. (usually 1.0 or 255)
        kernel_type: type of kernel, can be "gaussian" or "uniform".
        kernel_size: size of kernel
        kernel_sigma: standard deviation for Gaussian kernel.
        k1: stability constant used in the luminance denominator
        k2: stability constant used in the contrast denominator
        weights: parameters for image similarity and contrast sensitivity at different resolution scores.
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans)
    """

    def __init__(
        self,
        spatial_dims: int,
        data_range: float = 1.0,
        kernel_type: KernelType | str = KernelType.GAUSSIAN,
        kernel_size: int | Sequence[int, ...] = 11,
        kernel_sigma: float | Sequence[float, ...] = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        weights: Sequence[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)

        self.spatial_dims = spatial_dims
        self.data_range = data_range
        self.kernel_type = kernel_type

        if not isinstance(kernel_size, Sequence):
            kernel_size = ensure_tuple_rep(kernel_size, spatial_dims)
        self.kernel_size = kernel_size

        if not isinstance(kernel_sigma, Sequence):
            kernel_sigma = ensure_tuple_rep(kernel_sigma, spatial_dims)
        self.kernel_sigma = kernel_sigma

        self.k1 = k1
        self.k2 = k2
        self.weights = weights

    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted image.
                It must be a 2D or 3D batch-first tensor [B,C,H,W] or [B,C,H,W,D].
            y: Reference image.
                It must be a 2D or 3D batch-first tensor [B,C,H,W] or [B,C,H,W,D].

        Raises:
            ValueError: when `y_pred` is not a 2D or 3D image.
        """
        dims = y_pred.ndimension()
        if self.spatial_dims == 2 and dims != 4:
            raise ValueError(
                f"y_pred should have 4 dimensions (batch, channel, height, width) when using {self.spatial_dims} "
                f"spatial dimensions, got {dims}."
            )

        if self.spatial_dims == 3 and dims != 5:
            raise ValueError(
                f"y_pred should have 4 dimensions (batch, channel, height, width, depth) when using {self.spatial_dims}"
                f" spatial dimensions, got {dims}."
            )

        # check if image have enough size for the number of downsamplings and the size of the kernel
        weights_div = max(1, (len(self.weights) - 1)) ** 2
        y_pred_spatial_dims = y_pred.shape[2:]
        for i in range(len(y_pred_spatial_dims)):
            if y_pred_spatial_dims[i] // weights_div <= self.kernel_size[i] - 1:
                raise ValueError(
                    f"For a given number of `weights` parameters {len(self.weights)} and kernel size "
                    f"{self.kernel_size[i]}, the image height must be larger than "
                    f"{(self.kernel_size[i] - 1) * weights_div}."
                )

        weights = torch.tensor(self.weights, device=y_pred.device, dtype=torch.float)

        avg_pool = getattr(F, f"avg_pool{self.spatial_dims}d")

        multiscale_list: list[torch.Tensor] = []
        for _ in range(len(weights)):
            ssim, cs = compute_ssim_and_cs(
                y_pred=y_pred,
                y=y,
                spatial_dims=self.spatial_dims,
                data_range=self.data_range,
                kernel_type=self.kernel_type,
                kernel_size=self.kernel_size,
                kernel_sigma=self.kernel_sigma,
                k1=self.k1,
                k2=self.k2,
            )

            cs_per_batch = cs.view(cs.shape[0], -1).mean(1)

            multiscale_list.append(torch.relu(cs_per_batch))
            y_pred = avg_pool(y_pred, kernel_size=2)
            y = avg_pool(y, kernel_size=2)

        ssim = ssim.view(ssim.shape[0], -1).mean(1)
        multiscale_list[-1] = torch.relu(ssim)
        multiscale_list = torch.stack(multiscale_list)

        ms_ssim_value_full_image = torch.prod(multiscale_list ** weights.view(-1, 1), dim=0)

        ms_ssim_per_batch: torch.Tensor = ms_ssim_value_full_image.view(ms_ssim_value_full_image.shape[0], -1).mean(
            1, keepdim=True
        )

        return ms_ssim_per_batch


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
        default="/home/bru/PycharmProjects/DDPM-EEG/data/ids_shhs/ids_shhs_h_test.csv",
        #default="/home/bru/PycharmProjects/DDPM-EEG/data/ids/ids_sleep_edfx_cassette_double_test.csv",
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
        #default="/home/bru/PycharmProjects/DDPM-EEG/data/data_test",
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

    test_loader = test_dataloader(config=config_aekl, args=args, upper_limit=1000,)

    test_loader_2 = test_dataloader(config=config_aekl, args=args, upper_limit=1000,)

    ms_ssim = MultiScaleSSIMMetric(spatial_dims=1, data_range=1.0, kernel_size=7)


    print("Computing MS-SSIM...")
    ms_ssim_list = []
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for step, batch in pbar:
        x = batch['eeg'].to(device)
        for batch2 in test_loader_2:
            x2 = batch2["eeg"].to(device)
            if batch["eeg_meta_dict"]["filename_or_obj"][0] == batch2["eeg_meta_dict"]["filename_or_obj"][0]:
                continue
                
            x1_cropped = x[:,:,36:-36]
            x2_cropped = x2[:,:,36:-36]
            
            ms_ssim_list.append(ms_ssim(x1_cropped, x2_cropped).cpu())
        pbar.update()
        
    ms_ssim_list = torch.cat(ms_ssim_list, dim=0)
    print(f"Mean MS-SSIM: {ms_ssim_list.mean():.6f}")



if __name__ == "__main__":
    args = parse_args()
    main(args)
