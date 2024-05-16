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
    data = np.load('/data/esther_gi_tomo/ML-tomo/SampleG/sino_all_list_n1w1f1.npy')
    nslice, nang, px = data.shape
    data = remove_nan(data)
    theta = angles(nang, ang1=0, ang2=360)
    iter_num = 1500
    for slice in range(nslice):
        prj = data[slice]
        prj = nor_tomo(prj)
        prj = remove_nan(prj)
        if slice ==0:
            gan_tomo_object = GANtomo(prj, theta, iter_num=2000, save_wpath = '/data/weights/')
        else:
            gan_tomo_object = GANtomo(prj, theta, iter_num=800, init_wpath = '/data/weights/')
        start = time.time()
        rec = gan_tomo_object.recon
        end = time.time()
        print('Running time for slice {} is {}'.format(slice, end - start))
        fname_rec = '/data/esther_gi_tomo/ML-tomo//SampleG/reconc_ganrec/rec' + "-%03d" % (slice)
        tifffile.imwrite(fname_rec, rec.reshape((px, px)))
        plt.close()
if __name__ == "__main__":
    main()
