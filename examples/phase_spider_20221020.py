import time
import numpy as np
import dxchange
from ganrec.utils import nor_phase
from ganrec.ganrec2 import GANphase

def main():
    energy = 11
    z = 7.8880960e-2
    pv = 1.04735263e-7
    iter_num = 2000
    # abs_ratio = 0.06
    fname_data = '/data/gan_phase/data_spider.tif'
    data = dxchange.read_tiff(fname_data)
    nprj, px,  _ = data.shape
    data = nor_phase(data)
    
    abs_ratio_all = np.arange(0.01, 0.1, 0.01)
    for i, abs_ratio in enumerate(abs_ratio_all):
        gan_phase_object = GANphase(data[i], energy, z, pv, 
                                    abs_ratio = abs_ratio, 
                                    iter_num = iter_num,
                                    phase_only=False)
        start = time.time()
        absorption, phase = gan_phase_object.recon
        end = time.time()
        print('Running time is {}'.format(end - start))
        dxchange.write_tiff(absorption.reshape((px, px)), 
                            '/data/gan_phase/spider_abs_ratio/abs/absorption'"-%03d" % (i), 
                            overwrite=True)
        dxchange.write_tiff(phase.reshape((px, px)), 
                            '/data/gan_phase/spider_abs_ratio/phase/phase'"-%03d" % (i), 
                            overwrite=True)
          
    # for i in range(nprj):
    #     gan_phase_object = GANphase(data[i], energy, z, pv, 
    #                                 abs_ratio = abs_ratio, 
    #                                 iter_num = iter_num,
    #                                 phase_only=False)
    #     start = time.time()
    #     absorption, phase = gan_phase_object.recon
    #     end = time.time()
    #     print('Running time is {}'.format(end - start))
    #     dxchange.write_tiff(absorption.reshape((px, px)), 
    #                         '/data/gan_phase/spider_recon_20221020/abs/absorption'"-%03d" % (i), 
    #                         overwrite=True)
    #     dxchange.write_tiff(phase.reshape((px, px)), 
    #                         '/data/gan_phase/spider_recon_20221020/phase/phase'"-%03d" % (i), 
    #                         overwrite=True)

if __name__ == "__main__":
    main()