import numpy as np
import tifffile
import matplotlib.pyplot as plt
from ganrectf.utils import angles, nor_tomo
from ganrectf.ganrec import GANtensor


def main():
    prj = tifffile.imread('/nsls2/users/xyang4/data_tmp/tensor_tomography/strain_sino_tf_3.tiff')
    psi = 90
    psi = psi * np.pi / 180
    nang, px = prj.shape
    ang = angles(nang)
    prj = nor_tomo(prj)
    rec = GANtensor(prj, ang, psi, iter_num=6000).recon
    print(rec.shape)
    tifffile.imwrite('/nsls2/users/xyang4/data_tmp/tensor_tomography/strain_recon.tiff', rec)
if __name__ == "__main__":
    main()