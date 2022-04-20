from __future__ import  absolute_import, division, print_function
import numpy as np
import dxchange
import time
from xlearn.ganrec import rec_dcgan, rec_cost, angles
# from xlearn.utils import nor_data, nor_prj, center


def nor_data(img):
    # mean_tmp = np.mean(img)
    # std_tmp = np.std(img)
    # img = (img - mean_tmp) / std_tmp
    img = (img - img.min())/(img.max()-img.min())
    return img
def main():
    ###########################################################################################################
    sdir_ltp = '/data/xlearn_data/data_tomo/test_pattern_exp/presentation/full_angle'
    save_wpath = '/data/xlearn_data/weights_tomo/est.ckpt'
    ini_wpath = '/data/xlearn_data/data_tomo/test_pattern_exp/test_ini.ckpt'
    fname_data = '/data/xlearn_data/data_tomo/test_pattern_exp/prj256_181_2.tiff'
    data = dxchange.read_tiff(fname_data)
    nang, nslice, px = data.shape
    # ang = np.arange(nang)
    theta = angles(nang, ang1=0, ang2=180)
    slice = 100
    prj = data[:,slice,:]
    prj = nor_data(prj)
    prj = np.repeat(prj[np.newaxis, :, :, np.newaxis], 1, axis=0)
    theta = theta.astype('float32')
    prj = prj.astype('float32')
    start = time.time()
    recon = rec_dcgan(prj, theta, save_wpath, num_steps=2000, cost_rate=10, method='fc', learning_rate=1e-3)
    end = time.time()
    print('The prediction runs for %s seconds' % (end - start))
    sname_xbic = sdir_ltp + "_ganrec_5000"
    # dxchange.write_tiff(np.reshape(recon,(256,256)), sname_xbic, dtype='float32')
    # start = time.time()
    # recon = rec_cost(prj, theta, save_wpath, num_steps=5000, method='fc', learning_rate=1e-3)
    # end = time.time()
    # print('The prediction runs for %s seconds' % (end - start))
    # #     # recon = rec_dcgan(prj, theta, save_wpath, init_wpath=ini_wpath, num_steps=300, method='fc')
    # sname_xbic = sdir_ltp + "_dnn_5000"
    # dxchange.write_tiff(np.reshape(recon,(256,256)), sname_xbic, dtype='float32')
    # start = time.time()
    # recon = rec_cost(prj, theta, save_wpath, num_steps=5000, method='backproj', learning_rate=1e-2)
    # end = time.time()
    # print('The prediction runs for %s seconds' % (end - start))
    # # recon = rec_dcgan(prj, theta, save_wpath, init_wpath=ini_wpath, num_steps=300, method='fc')
    # # print(recon.dtype, recon.shape)
    # sname_xbic = sdir_ltp + "_bpcnn_5000"
    # dxchange.write_tiff(np.reshape(recon,(256,256)), sname_xbic, dtype='float32')

# def main():
#     ###########################################################################################################
#     sdir_ltp = '/gpfs/petra3/scratch/yangx/data_tomo/test_pattern_exp/rec256_181_test/'
#     save_wpath = '/gpfs/petra3/scratch/yangx/weights_tomo/est.ckpt'
#     ini_wpath = '/gpfs/petra3/scratch/yangx/data_tomo/test_pattern_exp/test_ini.ckpt'
#     fname_data = '/gpfs/petra3/scratch/yangx/data_tomo/test_pattern_exp/prj256_181_2.tiff'
#     data = dxchange.read_tiff(fname_data)
#     nang, nslice, px = data.shape
#     ang = np.arange(nang)
#     theta = np.asarray(ang) * np.pi / 180
#     for slice in range(nslice-32):
#         prj = data[:,slice,:]
#         prj = nor_data(prj)
#         prj = np.repeat(prj[np.newaxis, :, :, np.newaxis], 1, axis=0)
#         theta = theta.astype('float32')
#         prj = prj.astype('float32')
#         # recon = rec_dcgan(prj, theta, save_wpath, num_steps=2500, method='fc')
#         recon = rec_cost(prj, theta, save_wpath, num_steps=2500, method='fc')
#         # recon = rec_dcgan(prj, theta, save_wpath, init_wpath=ini_wpath, num_steps=300, method='fc')
#         sname_xbic = sdir_ltp + "-%03d" % (slice)
#         dxchange.write_tiff(recon, sname_xbic, dtype='float32')

if __name__ == "__main__":
    main()