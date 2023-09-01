import time
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from ganrec.utils import angles, nor_tomo, nor_prj
from ganrec.ganrec2 import GANtomo


def remove_nan(arr, val=0., ncore=None):

    arr[np.isnan(arr)]= 0
    return arr

def main():
    for i in range(5):
        data = tifffile.imread(f'/nsls2/data/staff/xyang4/data/esther_gi_tomo/star/sino_{i}.tiff')
        theta =angles(361)
        # prj = nor_tomo(data)
        gan_tomo_object = GANtomo(data, theta, iter_num=2500, save_wpath = '/nsls2/data/staff/xyang4/data/weights/')
        start = time.time()
        rec = gan_tomo_object.recon
        end = time.time()
        print('Running time for slice {} is {}'.format(slice, end - start))
        fname_rec = f'/nsls2/data/staff/xyang4/data/esther_gi_tomo/star/ganrec_{i}.tiff'
        tifffile.imwrite(fname_rec, rec)

if __name__ == "__main__":
    main()
