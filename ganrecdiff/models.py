"""Hugging Face ``diffusers`` based models for GANrec reconstruction.

The reconstruction generator is a :class:`diffusers.UNet2DModel` used as a
deterministic image-to-image network (fixed ``timestep=0``).  It maps a
measurement (sinogram / hologram / diffraction pattern) to the object being
reconstructed.  The forward physics model then re-projects the reconstruction
into measurement space where it is compared against the input.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diffusers import UNet2DModel
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "ganrecdiff requires the 'diffusers' package. "
        "Install it with: pip install diffusers"
    ) from exc


class DiffusersGenerator(nn.Module):
    """UNet2D generator backed by Hugging Face ``diffusers``.

    Parameters
    ----------
    out_h, out_w : int
        Spatial size of the reconstruction produced by the generator.
    in_channels : int, optional
        Number of input measurement channels (default 1).
    output_num : int, optional
        Number of reconstruction output channels (default 1).
    block_out_channels : tuple of int, optional
        Channel width of each UNet resolution level.  Its length sets the
        number of down/up-sampling stages.
    layers_per_block : int, optional
        Residual blocks per resolution level (default 2).
    """

    def __init__(
        self,
        out_h,
        out_w,
        in_channels=1,
        output_num=1,
        block_out_channels=(64, 128, 256, 256),
        layers_per_block=2,
    ):
        super().__init__()
        self.out_h = int(out_h)
        self.out_w = int(out_w)

        # Spatial dims fed to the UNet must be divisible by 2**(levels-1).
        self._divisor = 2 ** (len(block_out_channels) - 1)
        self._unet_h = int(np.ceil(self.out_h / self._divisor) * self._divisor)
        self._unet_w = int(np.ceil(self.out_w / self._divisor) * self._divisor)

        down_blocks = ["DownBlock2D"] * (len(block_out_channels) - 1) + ["AttnDownBlock2D"]
        up_blocks = ["AttnUpBlock2D"] + ["UpBlock2D"] * (len(block_out_channels) - 1)

        self.unet = UNet2DModel(
            sample_size=(self._unet_h, self._unet_w),
            in_channels=in_channels,
            out_channels=output_num,
            layers_per_block=layers_per_block,
            block_out_channels=tuple(block_out_channels),
            down_block_types=tuple(down_blocks),
            up_block_types=tuple(up_blocks),
        )

    def forward(self, x):
        # Resize the measurement onto the UNet grid.
        x = F.interpolate(x, size=(self._unet_h, self._unet_w), mode="bilinear", align_corners=False)
        t = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        out = self.unet(x, t).sample
        # Map the output back to the requested reconstruction resolution.
        if (self._unet_h, self._unet_w) != (self.out_h, self.out_w):
            out = F.interpolate(out, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False)
        return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Discriminator(nn.Module):
    """Convolutional discriminator operating in measurement space."""

    def __init__(self, in_channels=1):
        super().__init__()
        self.discriminator_model = nn.Sequential(
            nn.Conv2d(in_channels, 16, (5, 5), stride=(2, 2)),
            nn.Conv2d(16, 16, (5, 5), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(16, 32, (5, 5), stride=(2, 2)),
            nn.Conv2d(32, 32, (5, 5), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, (3, 3), stride=(2, 2)),
            nn.Conv2d(64, 64, (3, 3), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, (3, 3), stride=(2, 2)),
            nn.Conv2d(128, 128, (3, 3), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            Flatten(),
            nn.LazyLinear(512),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.discriminator_model(x)
