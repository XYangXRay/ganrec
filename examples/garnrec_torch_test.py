
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from ganrectorch.utils import angles, nor_tomo
from ganrectorch.ganrec import GANtomo

prj = tifffile.imread('./test_data/tooth.tiff')
nang, px = prj.shape
ang = angles(nang)
prj = nor_tomo(prj)
rec = GANtomo(prj, ang, iter_num=1000).recon()
plt.imshow(rec)
plt.show()
tifffile.imwrite('./test_results/recon_tooth', rec)