import numpy as np
import tifffile
import matplotlib.pyplot as plt
from ganrec.utils import angles, nor_tomo
from ganrec.ganrec2 import GANtomo


def main():
    prj = tifffile.imread('./test_data/tooth.tiff')
    nang, px = prj.shape
    ang = angles(nang)
    prj = nor_tomo(prj)
    gan_tomo_object = GANtomo(prj, ang, iter_num=2000)
    rec = gan_tomo_object.recon
    tifffile.imwrite('./test_results/tooth_recon.tiff', rec)
if __name__ == "__main__":
    main()