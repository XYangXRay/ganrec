import time
import numpy as np
import tifffile
import os
from ganrec.utils import nor_phase
from ganrec.ganrec2 import GANphase

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

def main():
    energy = 15
    # z = 2e-3
    z = 0.889e-3
    pv =1*6.1e-8
    # pv = 55e-6
    iter_num = 3000
    # abs_ratio = 0.06
    fname_data = '/nsls2/data/staff/xyang4/data/hanfei_phase/1mm_001.tiff'
    data = tifffile.imread(fname_data)
    px, px= data.shape
    data = nor_phase(data)
    # data_tmp = data[50:250, 50:250]

        
    gan_phase_object = GANphase(data, energy, z, pv, 
                                    abs_ratio = 0.1, 
                                    iter_num = iter_num,
                                    phase_only=False)
    start = time.time()
    absorption, phase = gan_phase_object.recon
    end = time.time()
    print('Running time is {}'.format(end - start))
  
    save_tiff(absorption.reshape((px, px)), 
                        '/nsls2/data/staff/xyang4/data/hanfei_phase/results/1mm_001_absorption.tiff')
    save_tiff(phase.reshape((px, px)), 
                        '/nsls2/data/staff/xyang4/data/hanfei_phase/results/1mm_001_phase.tiff')

if __name__ == "__main__":
    main()