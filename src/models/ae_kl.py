"""
AUTOENCODER WITH ARCHTECTURE FROM VERSION 2
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def swish(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        #print("in_channels", in_channels)
        self.conv = nn.Conv1d(
            in_channels, in_channels, kernel_size=3, padding=0, stride=2
        )

    def forward(self, x):
        pad = (0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = Normalize(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv1d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h = q.shape
        q = q.reshape(b, c, h )
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h )  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h )
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_channels: int,
        z_channels: int,
        ch_mult: Tuple[int],
        num_res_blocks: int,
        resolution: Tuple[int],
        attn_resolutions: Tuple[int],
        **ignorekwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_channels = n_channels
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)

        blocks = []
        # initial convolution
        blocks.append(
            nn.Conv1d(in_channels, n_channels, kernel_size=3, stride=1, padding=1)
        )

        # residual and downsampling blocks,
        # with attention on smaller res (16x16)
        for i in range(self.num_resolutions):
            block_in_ch = n_channels * in_ch_mult[i]
            block_out_ch = n_channels * ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if max(curr_res) in attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != self.num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = tuple(ti // 2 for ti in curr_res)

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        # normalise and convert to latent size
        blocks.append(Normalize(block_in_ch))
        blocks.append(
            nn.Conv1d(block_in_ch, z_channels, kernel_size=3, stride=1, padding=1)
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        z_channels: int,
        out_channels: int,
        ch_mult: Tuple[int],
        num_res_blocks: int,
        resolution: Tuple[int],
        attn_resolutions: Tuple[int],
        **ignorekwargs,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.z_channels = z_channels
        self.out_channels = out_channels
        self.ch_mult = ch_mult
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions

        block_in_ch = n_channels * self.ch_mult[-1]
        curr_res = tuple(ti // 2 ** (self.num_resolutions - 1) for ti in resolution)

        blocks = []
        # initial conv
        blocks.append(
            nn.Conv1d(z_channels, block_in_ch, kernel_size=3, stride=1, padding=1)
        )

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = n_channels * self.ch_mult[i]

            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

                if max(curr_res) in self.attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = tuple(ti * 2 for ti in curr_res)

        blocks.append(Normalize(block_in_ch))
        blocks.append(
            nn.Conv1d(block_in_ch, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class AutoencoderKL(nn.Module):
    def __init__(self, embed_dim: int, hparams) -> None:
        super().__init__()
        self.encoder = Encoder(**hparams)
        self.decoder = Decoder(**hparams)
        self.quant_conv_mu = torch.nn.Conv1d(hparams["z_channels"], embed_dim, 1)
        self.quant_conv_log_sigma = torch.nn.Conv1d(hparams["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv1d(embed_dim, hparams["z_channels"], 1)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)

        z_mu = self.quant_conv_mu(h)
        z_log_var = self.quant_conv_log_sigma(h)
        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var / 2)

        return z_mu, z_sigma

    def sampling(self, z_mu, z_sigma):
        eps = torch.randn_like(z_sigma)
        z_vae = z_mu + eps * z_sigma
        return z_vae

    def reconstruct(self, x):
        z_mu, _ = self.encode(x)
        reconstruction = self.decode(z_mu)
        return reconstruction

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x, get_ldm_inputs=False):
        if get_ldm_inputs:
            return self.get_ldm_inputs(x)
        else:
            z_mu, z_sigma = self.encode(x)
            z = self.sampling(z_mu, z_sigma)
            reconstruction = self.decode(z)
            return reconstruction, z_mu, z_sigma

    def get_ldm_inputs(self, img):
        z_mu, z_sigma = self.encode(img)
        z = self.sampling(z_mu, z_sigma)
        return z

    def reconstruct_ldm_outputs(self, z):
        x_hat = self.decode(z)
        return x_hat


class VAE_downsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.MaxPool1d(kernel_size=4, stride=4)
        self.transposed_model = nn.Upsample(scale_factor=4, mode="bilinear")

    def forward(self, x, get_ldm_inputs=True):
        if get_ldm_inputs:
            return self.get_ldm_inputs(x)
        else:
            z = self.model(x)
            return self.reconstruct_ldm_outputs(z)

    def get_ldm_inputs(self, img):
        return self.model(img)

    def reconstruct_ldm_outputs(self, img):
        return self.transposed_model(img)
