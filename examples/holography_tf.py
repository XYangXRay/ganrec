import time
import tifffile
from ganrectf.utils import nor_phase
from ganrectf.ganrec import GANphase


def main():
    energy = 10
    z = 0.5
    pv = 5e-7
    iter_num = 1000
    fname_data = "./test_data/hologram_shepp.tiff"
    data = tifffile.imread(fname_data)
    px, _ = data.shape
    data = nor_phase(data)
    start = time.time()
    abs, phase = GANphase(data, energy, z, pv, iter_num=iter_num).recon
    end = time.time()
    print("Running time is {}".format(end - start))
    tifffile.imwrite("./test_results/phase_shepp.tiff", phase)


if __name__ == "__main__":
    main()
