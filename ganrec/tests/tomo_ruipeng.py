import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path
import os
from ganrec.utils import angles, nor_tomo
from ganrec.ganrec2 import GANtomo

fpath = '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/crop_bin4_recon_pad_20240416/'
spath = '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/crop_bin4_tomo_20240417/'

def list_files(directory):
    path = Path(directory)
    for file_path in path.rglob('*'):
        yield file_path
        
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
    
def nor_diff(img):
 
    # img = np.log(img+2)
    mean_tmp = np.mean(img)
    std_tmp = np.std(img)
    img = (img - mean_tmp) / std_tmp
    img = (img - img.min()) / (img.max() - img.min())
    return img

def nor_prj(img):
    # nang, px = img.shape
    mean_sum = np.mean(np.sum(img, axis=(1,2)))
    data_corr = np.zeros_like(img)
    for i in range(len(img)):
        data_corr[i, :, :] = img[i, :, :] * mean_sum / np.sum(img[i, :, :])
    return data_corr
# data_ind = [1, 2, 3, 34, 35, 36, 47, 48, 49, 57, 58, 59, 73, 74, 75, 88, 89, 90, 
#             100, 101, 102, 114, 115, 116, 144, 145, 146, 173, 174, 175, 204, 205, 
#             206, 232, 233, 234, 255, 256, 257, 305, 306, 307, 336, 337, 338]
data_ind = [2, 3, 4, 5, 6, 7, 8, 9, 10, 31, 32, 33, 34, 35, 36, 37, 38, 46, 49, 57, 
            58, 59, 60, 71, 72, 73, 74, 75, 77, 79, 81, 82, 83, 88, 89, 90, 97, 98, 
            100, 101, 103, 104, 106, 107, 109, 110, 112, 113, 114, 115, 116, 117, 118,
            119, 121, 122, 123, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148,
            149, 150, 151, 152, 153, 154, 155, 171, 173, 176, 177, 178, 179, 180, 202, 
            203, 204, 205, 206, 207, 208, 216, 217, 219, 221, 223, 225, 226, 228, 229, 
            230, 231, 232, 233, 235, 250, 251, 252, 253, 255, 256, 257, 258, 260, 261,
            286, 288, 293, 294, 296, 297, 298, 300, 301, 306, 307, 308, 335, 336, 339, 
            340]
updated_ind = [x - 1 for x in data_ind]
# print(updated_ind)
data = []
for file_name in list_files(fpath):
    data_tmp = tifffile.imread(file_name)
    data.append(data_tmp)
data = np.array(data)
print(data.shape)

theta = angles(361, ang1=0, ang2=180)
theta_select = theta[updated_ind]

iter_num = 1500
# data = nor_prj(data)
# data_pad = np.zeros((361, 256, 364))
# data_pad[:, :, 54:-54] = data
# data[:, :, :32] = 0
# data[:, :, -32:] = 0
data_select = data[updated_ind]
# data_select = nor_prj(data_select)


for i in range(256):
    # prj_select = data_select[:, i+100,:]
  
    # gan_tomo_object = GANtomo(prj_select, theta_select, 
    #                       iter_num= iter_num,  l1_ratio = 200,
    #                       g_learning_rate = 5e-5, d_learning_rate = 5e-7)
    prj = data[:, i+100,:]
  
    gan_tomo_object = GANtomo(prj, theta, 
                          iter_num= iter_num,  l1_ratio = 200,
                          g_learning_rate = 5e-3, d_learning_rate = 5e-6)
    
    rec = gan_tomo_object.recon
    save_tiff(rec, spath+f'{i:03d}'+'.tiff')
    
    
    
    
    
    
# prj = data_pad[:, 120,:]
# prj_select = prj[data_ind]
# gan_tomo_object = GANtomo(prj_select, theta_select, 
#                           iter_num= iter_num,  l1_ratio = 200,
#                           g_learning_rate = 5e-5, d_learning_rate = 5e-7)
# rec = gan_tomo_object.recon
# save_tiff(rec, spath+os.path.splitext(file_name)[0][-6:]+'.tiff')