import numpy as np
import tifffile
from ganrectf.utils import angles
from ganrectf.ganrec import GANtensor


def main():
    prj = tifffile.imread("/nsls2/users/xyang4/data_tmp/tensor_tomography/strain_sino_tf_3.tiff")
    psi = 90
    psi = psi * np.pi / 180
    nang, px = prj.shape
    ang = angles(nang)
    rec = GANtensor(prj, ang, psi, iter_num=6000).recon
    tifffile.imwrite("/nsls2/users/xyang4/data_tmp/tensor_tomography/strain_recon.tiff", rec)


if __name__ == "__main__":
    main()
