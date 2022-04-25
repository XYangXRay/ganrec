import numpy as np
import matplotlib.pyplot as plt

def nor_data(img):
    mean_tmp = np.mean(img)
    std_tmp = np.std(img)
    img = (img - mean_tmp) / std_tmp
    img = (img - img.min()) / (img.max() - img.min())
    return img

def angles(nang, ang1=0., ang2=180.):
    return np.linspace(ang1 * np.pi / 180., ang2 * np.pi / 180., nang)

class RECONmonitor:
    def __init__(self):
        self.fig, self.axs = plt.subplots(2, 2, figsize=(16, 8))

    def initial_plot(self, img_input):
        _, px = img_input.shape
        self.im0 = self.axs[0, 0].imshow(img_input, cmap='gray')
        self.axs[0, 0].set_title('Sinogram')
        self.fig.colorbar(self.im0, ax=self.axs[0, 0])
        self.im1 = self.axs[1, 0].imshow(img_input, cmap='jet')
        self.tx1 = self.axs[1, 0].set_title('Difference of sinogram for iteration 0')
        self.fig.colorbar(self.im1, ax=self.axs[1, 0])
        self.im2 = self.axs[0, 1].imshow(np.zeros((px, px)), cmap='gray')
        self.fig.colorbar(self.im2, ax=self.axs[0, 1])
        self.axs[0, 1].set_title('Reconstruction')
        self.im3, = self.axs[1, 1].plot([], [], 'r-')
        self.axs[1, 1].set_title('Generator loss')
        plt.tight_layout()

    def update_plot(self, epoch, img_diff, img_rec, plot_x, plot_loss):
        self.tx1.set_text('Difference of sinogram for iteration {0}'.format(epoch))
        vmax = np.max(img_diff)
        vmin = np.min(img_diff)
        self.im1.set_data(img_diff)
        self.im1.set_clim(vmin, vmax)
        self.im2.set_data(img_rec)
        vmax = np.max(img_rec)
        vmin = np.min(img_rec)
        self.im2.set_clim(vmin, vmax)
        self.axs[1, 1].plot(plot_x, plot_loss, 'r-')
        # self.im3.set_xdata(plot_x)
        # self.im3.set_ydata(plot_loss)
        plt.pause(0.1)
