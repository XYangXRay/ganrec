import time
import os
from pathlib import Path
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tifffile
from ganrec.ganrec2 import GANdiffraction


fpath = '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/crop_bin/'
spath = '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/crop_bin_recon_20231220/'
fname_mask = '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/mask_crop_nor.tiff'
iter_num = 1200

mask = tifffile.imread(fname_mask)
def nor_diff(img):
 
    # img = np.log(img+2)
    mean_tmp = np.mean(img)
    std_tmp = np.std(img)
    img = (img - mean_tmp) / std_tmp
    img = (img - img.min()) / (img.max() - img.min())
    return img

def save_tiff(image, filename):
    # Extract the directory from the filename
    directory = os.path.dirname(filename)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)
    image = nor_diff(image)
    image = np.array(image, dtype = np.float32)
    # Save the image
    tifffile.imwrite(filename, image)


def list_files(directory):
    path = Path(directory)
    for file_path in path.rglob('*'):
        yield file_path
def main():
    # for i, file_name in enumerate(list_files(fpath)):
    for file_name in list_files(fpath):
        print("Reconstruction for " + str(file_name))
        data = tifffile.imread(file_name)
        px, _ = data.shape
        data = nor_diff(data)
        gan_diff_object = GANdiffraction(data, mask, iter_num=iter_num, recon_monitor=True)
        start = time.time()
        abs, rec = gan_diff_object.recon
        end = time.time()
        print('Running time is {}'.format(end - start))
        save_tiff(rec.reshape((px, px)), spath+str(file_name)[-6:]+'.tif')
        
    
    
# def main():

#     iter_num = 2000
#     # fname_data = '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/data_crop_nor.tiff'
#     fname_data = '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/crop_bin/796720'
#     fname_mask = '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/mask_crop_nor.tiff'
#     data = dxchange.read_tiff(fname_data)
#     # data  = data[128-100:128+100, 128-100:128+100]
#     mask = dxchange.read_tiff(fname_mask)
#     px, _ = data.shape
#     data = nor_diff(data)
#     gan_diff_object = GANdiffraction(data, mask, iter_num=iter_num)
#     start = time.time()
#     abs, rec = gan_diff_object.recon
#     end = time.time()
#     print('Running time is {}'.format(end - start))
#     print(rec.shape, rec.dtype)
#     dxchange.write_tiff(rec.reshape((px, px)), '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/crop/results/phase_log50', overwrite=True)
#     dxchange.write_tiff(abs.reshape((px, px)), '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/crop/results/abs_log50', overwrite=True)
#     # dxchange.write_tiff(rec.reshape((px, px)), '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/results/phase_loge3', overwrite=True)
#     # dxchange.write_tiff(abs.reshape((px, px)), '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/results/abs_loge3', overwrite=True)



if __name__ == "__main__":
    
    main()