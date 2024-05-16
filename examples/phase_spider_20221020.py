import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import tifffile
from ganrec.utils import nor_phase
from ganrec.ganrec2 import GANphase

def main():
    energy = 11
    z = 7.8880960e-2
    pv = 1.04735263e-7
    iter_num = 2000
    # abs_ratio = 0.06
    fname_data = '/data/gan_phase/data_spider.tif'
    data = tifffile.imread(fname_data)
    nprj, px,  _ = data.shape
    data = nor_phase(data)

          
if __name__ == "__main__":
    main()