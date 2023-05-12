import time
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import dxchange
# from ganrec.utils import nor_tomo
from ganrec.ganrec2 import GANdiffraction


def nor_diff(img):
 
    # img = np.log(img+2)
    mean_tmp = np.mean(img)
    std_tmp = np.std(img)
    img = (img - mean_tmp) / std_tmp
    img = (img - img.min()) / (img.max() - img.min())
    return img

def main():

    iter_num = 6000
    # fname_data = '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/data_crop_nor.tiff'
    fname_data = '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/test-2.tif'
    fname_mask = '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/mask_crop_nor.tiff'
    data = dxchange.read_tiff(fname_data)
    # data  = data[128-100:128+100, 128-100:128+100]
    mask = dxchange.read_tiff(fname_mask)
    px, _ = data.shape
    data = nor_diff(data)
    gan_diff_object = GANdiffraction(data, mask, iter_num=iter_num)
    start = time.time()
    abs, rec = gan_diff_object.recon
    end = time.time()
    print('Running time is {}'.format(end - start))
    print(rec.shape, rec.dtype)
    dxchange.write_tiff(rec.reshape((px, px)), '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/crop/results/phase_loge3', overwrite=True)
    dxchange.write_tiff(abs.reshape((px, px)), '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/crop/results/abs_loge3', overwrite=True)
    # dxchange.write_tiff(rec.reshape((px, px)), '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/results/phase_loge3', overwrite=True)
    # dxchange.write_tiff(abs.reshape((px, px)), '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/results/abs_loge3', overwrite=True)



if __name__ == "__main__":
    
    main()