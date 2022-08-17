import time
import numpy as np
import dxchange
import matplotlib.pyplot as plt
from ganrec.utils import angles, nor_tomo, nor_prj
from ganrec.ganrec2 import GANtomo

#
def remove_nan(arr, val=0., ncore=None):

    arr[np.isnan(arr)]= 0
    return arr

def main():
    fname_data = '/data/xrf_tomo_andy/exp3_Ni_mat_aligned.tiff'
    data_ = dxchange.read_tiff(fname_data)
    data_ = remove_nan(data_)
    data = nor_tomo(data_)
    # data = nor_prj(data_)
    # nang, px = data.shape
    nang, nslice, px = data.shape
    theta = angles(nang, ang1=0, ang2=180)
    # data = data_[1:, 10:-10, 20:-20]
    print(data.shape)


    # data = remove_nan(data)

    # theta = angles(nang, ang1=10, ang2=180)
    # theta = np.load('/data/xrf_tomo_andy/ang_exp2.npy')
    for slice in range(nslice):
        prj = data[:,slice,:]
        # prj = nor_tomo(prj)
        # prj = nor_prj(prj)
        # prj = remove_nan(prj)
        # gan_tomo_object = GANtomo(prj, theta, iter_num=400, init_wpath='/data/weights/')
        gan_tomo_object = GANtomo(prj, theta, iter_num=800)
        # if slice ==0:
        #     gan_tomo_object = GANtomo(prj, theta, iter_num=1000, save_wpath = '/data/weights/')
        # else:
        #     gan_tomo_object = GANtomo(prj, theta, iter_num=400, init_wpath = '/data/weights/')
        start = time.time()
        rec = gan_tomo_object.recon
        # prj_filter = prj_filter.numpy()
        end = time.time()
        print('Running time for slice {} is {}'.format(slice, end - start))
        # fname_rec = '/data/esther_gi_tomo/ML-tomo/rec_G_test/rec' + "-%03d" % (slice)
        fname_rec = '/data/xrf_tomo_andy/exp3_Ni_K_rec_20220803/recon' + "-%03d" % (slice)

        # fname_prj = '/data/esther_gi_tomo/ML-tomo/rec_G_test/prj' + "-%03d" % (slice)
        rec = remove_nan(rec)
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
