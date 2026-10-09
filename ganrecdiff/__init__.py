"""GANrec with Hugging Face ``diffusers`` UNet models as the generator.

Provides :class:`GANtomo`, :class:`GANphase`, and a generic :class:`GANrec`
reconstruction driver that use :class:`diffusers.UNet2DModel` in place of the
hand-written CNN generator.
"""

__author__ = """Xiaogang Yang"""
__email__ = "yangxg@bnl.gov"
__version__ = "0.1.0"

from ganrecdiff.ganrec import GANtomo, GANphase, GANrec
from ganrecdiff.models import DiffusersGenerator, Discriminator

__all__ = ["GANtomo", "GANphase", "GANrec", "DiffusersGenerator", "Discriminator"]
