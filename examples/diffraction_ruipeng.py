import time
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# import tomopy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tifffile
from ganrec.ganrec2 import GANdiffraction


fpath = '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/crop_20240624/'
spath = '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/crop64_recon_20250623/'
# spath = '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/crop_bin4_20240208/'
fname_mask = '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/mask_crop_nor.tiff'
iter_num = 800

# mask = tifffile.imread(fname_mask)

recon_list = [105, 135, 173, 214, 244, 297, 307, 323, 355, 386, 435]

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

def draw_mask(img, inner_diameter, outer_diameter):
    image_size, _ = img.shape
    x = np.linspace(-image_size // 2, image_size // 2, image_size)
    y = np.linspace(-image_size // 2, image_size // 2, image_size)
    X, Y = np.meshgrid(x, y)

    # Calculate distances from the center
    center = (0, 0)
    distances = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    # Create the mask
    mask = (distances >= inner_diameter / 2) & (distances <= outer_diameter / 2)

    # Apply the mask to an image (white ring on black background)
    img = img*mask

    return img

def make_mask(image_size, inner_diameter, outer_diameter):
    # image_size, _ = img.shape
    x = np.linspace(-image_size // 2, image_size // 2, image_size)
    y = np.linspace(-image_size // 2, image_size // 2, image_size)
    X, Y = np.meshgrid(x, y)

    # Calculate distances from the center
    center = (0, 0)
    distances = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    # Create the mask
    mask = (distances >= inner_diameter / 2) & (distances <= outer_diameter / 2)

    # Apply the mask to an image (white ring on black background)
    mask = np.ones((image_size, image_size))*mask
    # img = img*mask

    return np.float32(mask)

mask = make_mask(1024, 100, 980) 
print(mask.dtype)
data_all = tifffile.imread('/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/crop_bin4_20240208.tif')

def main():
    for i in range(361):
        file_name = fpath + str(796815+i) +'.tiff'
        # file_name = '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/test_1.tif'
        # file_name = fpath + str(796815+i)
        print("Reconstruction for " + str(file_name))
        # if i-1 in recon_list or i in recon_list or i+1 in recon_list:
            
        data = tifffile.imread(file_name)
        # save_tiff(data, spath+os.path.splitext(file_name)[0][-6:]+'.tiff')
        # data = data_all[i]
        data = draw_mask(data, 150, 980)
            # save_tiff(data, '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/test_tmp.tiff')
            # plt.imshow(data)
            # plt.show()
        px, _ = data.shape
        data = nor_diff(data)
        
        gan_diff_object = GANdiffraction(data, mask, iter_num=iter_num, recon_monitor=True)
        start = time.time()
        abs, rec = gan_diff_object.recon
        end = time.time()
        print('Running time is {}'.format(end - start))
        print(spath+os.path.splitext(file_name)[0][-6:]+'.tiff')
        save_tiff(rec.reshape((px//8, px//8)), spath+os.path.splitext(file_name)[0][-6:]+'.tiff')
# avg_num = 20
# def main():
#     for i in range(361):
#         file_name = fpath + str(796815+i) +'.tiff'

#         data = data_all[i]
#         data = draw_mask(data, 36, 220)

#         px, _ = data.shape
#         data = nor_diff(data)
#         start = time.time()
#         rec = np.zeros((avg_num, px, px))
#         for m in range(avg_num):
#             gan_diff_object = GANdiffraction(data, mask, iter_num=iter_num, recon_monitor=False, 
#                                          save_wpath= '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/weights/')
#             abs, rec_tmp = gan_diff_object.recon
#             rec[m, :, :] = rec_tmp.reshape((px, px))
#         end = time.time()
#         print('Running time is {}'.format(end - start))
#             # print(spath+os.path.splitext(file_name)[0][-6:]+'.tiff')
#         save_tiff(rec, spath+os.path.splitext(file_name)[0][-6:]+'.tiff')
#         save_tiff(np.mean(rec, axis=0), spath1+os.path.splitext(file_name)[0][-6:]+'.tiff')
  

# def main():
#     for i, file_name in enumerate(list_files(fpath)):
#         if i<360:
#         # if i-1 in recon_list or i in recon_list or i+1 in recon_list:
#             print("Reconstruction for " + str(file_name))
#             data = tifffile.imread(file_name)
#             data = draw_mask(data, 40, 240)
#             # save_tiff(data, '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/test_tmp.tiff')
#             # plt.imshow(data)
#             # plt.show()
#             px, _ = data.shape
#             data = nor_diff(data)
#             gan_diff_object = GANdiffraction(data, mask, iter_num=iter_num, recon_monitor=True)
#             start = time.time()
#             abs, rec = gan_diff_object.recon
#             end = time.time()
#             print('Running time is {}'.format(end - start))
#             # print(spath+os.path.splitext(file_name)[0][-6:]+'.tiff')
#             save_tiff(rec.reshape((px, px)), spath+os.path.splitext(file_name)[0][-6:]+'.tiff')
    # for file_name in list_files(fpath):
    
        
        # data = tifffile.imread(file_name)
        # px, _ = data.shape
        # data = nor_diff(data)
        # gan_diff_object = GANdiffraction(data, mask, iter_num=iter_num, recon_monitor=True)
        # start = time.time()
        # abs, rec = gan_diff_object.recon
        # end = time.time()
        # print('Running time is {}'.format(end - start))
        # save_tiff(rec.reshape((px, px)), spath+str(file_name)[:6]+'.tif')
        
    
    
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