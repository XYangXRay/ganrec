import numpy as np
import tifffile
import matplotlib.pyplot as plt
from ganrectorch.utils import angles, nor_tomo
from ganrectorch.ganrec import GANtomo

prj = tifffile.imread("./test_data/shale_prj.tiff")
nang, px = prj.shape
ang = angles(nang)
prj = nor_tomo(prj)
rec = GANtomo(prj, ang, iter_num=1000).recon()
tifffile.imwrite("./test_results/recon_shale.tiff", rec)
