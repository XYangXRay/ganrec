import time
import dxchange
from ganrec.utils import angles, nor_tomo
from ganrec.ganrec2 import GANtomo

def main():
    fname_data = 'tooth.tiff'
    data = dxchange.read_tiff(fname_data)
    nang, px = data.shape
    theta = angles(nang, ang1=0, ang2=180)
    # slice = 100
    iter_num = 1000
    # prj = data[:, slice, :]
    prj = nor_tomo(data)
    gan_tomo_object = GANtomo(prj, theta, iter_num)
    start = time.time()
    rec = gan_tomo_object.recon
    end = time.time()
    print('Running time is {}'.format(end - start))
    dxchange.write_tiff(rec.reshape((px, px)), '/data/ganrec/test_filter', overwrite=True)


if __name__ == "__main__":
    main()