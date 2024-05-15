import time
import numpy as np
import dxchange
from ganrec.utils import nor_phase
from ganrec.ganrec2 import GANphase

def main():
    energy = 12
    # z = 2.9878e-2
    # z = 1.057
    z = 2.9878e-2
    pv=3.88e-8
    # pv =5.52e-6/36.5
    pv = 1.38e-6
    iter_num = 1400
    # abs_ratio = 0.06
    fname_data = '/nsls2/data/staff/xyang4/data/phase_yang/dist3-1.tif'
    data = dxchange.read_tiff(fname_data)
    px, px= data.shape
    data = nor_phase(data)
    # for i in range(10):
    #     z = pv/(2*(i+1))
    #     gan_phase_object = GANphase(data, energy, z, pv, 
    #                                 abs_ratio = 0.4, 
    #                                 iter_num = iter_num,
    #                                 phase_only=False)
    #     start = time.time()
    #     absorption, phase = gan_phase_object.recon
    #     end = time.time()
    #     print('Running time is {}'.format(end - start))
    #     dxchange.write_tiff(absorption.reshape((px, px)), 
    #                     f'/data/phase_yang/absorption_{z}', 
    #                     overwrite=True)
    #     dxchange.write_tiff(phase.reshape((px, px)), 
    #                     f'/data/phase_yang/phase_{z}', 
    #                     overwrite=True)
        
    gan_phase_object = GANphase(data, energy, z, pv, 
                                    abs_ratio = 0.1, 
                                    iter_num = iter_num,
                                    phase_only=False)
    start = time.time()
    absorption, phase = gan_phase_object.recon
    end = time.time()
    print('Running time is {}'.format(end - start))
    dxchange.write_tiff(absorption.reshape((px, px)), 
                        '/nsls2/data/staff/xyang4/data/phase_yang/absorption_', 
                        overwrite=True)
    dxchange.write_tiff(phase.reshape((px, px)), 
                        '/nsls2/data/staff/xyang4/data/phase_yang/phase_', 
                        overwrite=True)
    # for i in range(20):
        

if __name__ == "__main__":
    main()