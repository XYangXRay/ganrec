import time
import numpy as np
import dxchange
import matplotlib.pyplot as plt
from ganrec.utils import angles, nor_tomo, nor_prj
from ganrec.ganrec2 import GANtomo


def remove_nan(arr, val=0., ncore=None):

    arr[np.isnan(arr)]= 0
    return arr

def main():
    # fname_data = 'tooth.tiff'
    # data = dxchange.read_tiff(fname_data)
    # nang, px = data.shape
    # theta = angles(nang, ang1=0, ang2=180)
    data = np.load('/data/esther_gi_tomo/ML-tomo/SampleS/sino_all_list_n1w1f1.npy')
    # data = np.load('/data/esther_gi_tomo/ML-tomo/SampleG/sino_all_list.npy')
    nslice, nang, px = data.shape
    # theta = np.load('/data/esther_gi_tomo/ML-tomo/SampleG/domain_angle_offset.npy', allow_pickle=True)
    data = remove_nan(data)
    # slice = 100 
    theta = angles(nang, ang1=0, ang2=360)
    iter_num = 1500
    for slice in range(nslice):
        prj = data[slice]
        # prj = nor_tomo(prj)
        # prj = nor_prj(prj)
        if slice ==0:
            gan_tomo_object = GANtomo(prj, theta, iter_num=2000, save_wpath = '/data/weights/')
        else:
            gan_tomo_object = GANtomo(prj, theta, iter_num=800, init_wpath = '/data/weights/')
        start = time.time()
        rec = gan_tomo_object.recon
        # prj_filter = prj_filter.numpy()
        end = time.time()
        print('Running time for slice {} is {}'.format(slice, end - start))
        # fname_rec = '/data/esther_gi_tomo/ML-tomo/rec_G_test/rec' + "-%03d" % (slice)
        fname_rec = '/data/esther_gi_tomo/ML-tomo//SampleS/reconc_ganrec/rec' + "-%03d" % (slice)

        # fname_prj = '/data/esther_gi_tomo/ML-tomo/rec_G_test/prj' + "-%03d" % (slice)
        dxchange.write_tiff(rec.reshape((px, px)), fname_rec, overwrite=True, dtype='float32')
        # dxchange.write_tiff(prj_filter.reshape((nang, px)), fname_prj, overwrite=True)
        plt.close()

    # prj = data[100, :, :]
    # prj = nor_tomo(prj)
    # gan_tomo_object = GANtomo(prj, theta, iter_num)
    # start = time.time()
    # rec = gan_tomo_object.recon
    # end = time.time()
    # print('Running time is {}'.format(end - start))
    # dxchange.write_tiff(rec.reshape((px, px)), '/nsls2/users/xyang4/data/rec/100', overwrite=True)


if __name__ == "__main__":
    main()
