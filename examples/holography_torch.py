import tifffile
import matplotlib.pyplot as plt
from ganrectorch.utils import nor_tomo
from ganrectorch.ganrec import GANphase
energy = 11
z = 7.8880960e-2
pv = 1.04735263e-7
abs_ratio = 0.06
iter_num = 2000
g_learning_rate = 1e-3
d_learning_rate = 1e-6
fname_data = './test_data/hologram_spider_hair.tiff'
data = tifffile.imread(fname_data)
data = nor_tomo(data)
abs, phase = GANphase(data, energy, z, pv, iter_num=iter_num).recon()
plt.imshow(phase)
plt.show()
tifffile.imwrite('./test_results/phase_shepp.tiff', phase)