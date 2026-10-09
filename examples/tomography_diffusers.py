"""Tomographic reconstruction using a Hugging Face diffusers UNet generator.

Run from the ``examples/`` directory:

    python tomography_diffusers.py
"""

import numpy as np
import tifffile

from ganrectorch.utils import angles, nor_tomo
from ganrecdiff.ganrec import GANtomo

prj = tifffile.imread("./test_data/shale_prj.tiff")
nang, px = prj.shape
ang = angles(nang)
prj = nor_tomo(prj)

# UNet width/depth is controlled by ``block_out_channels``; more/wider levels
# increase capacity at the cost of memory.
rec = GANtomo(
    prj,
    ang,
    iter_num=1000,
    block_out_channels=[64, 128, 256, 256],
    g_learning_rate=1e-4,
).recon()

tifffile.imwrite("./test_results/recon_shale_diffusers.tiff", np.squeeze(rec))
