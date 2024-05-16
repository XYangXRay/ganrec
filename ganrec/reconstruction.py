def main():
    for i in range(361):
        file_name = fpath + str(796815+i) +'.tiff'

        data = data_all[i]
        data = draw_mask(data, 36, 220)
            # save_tiff(data, '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/test_tmp.tiff')
            # plt.imshow(data)
            # plt.show()
        px, _ = data.shape
        data = nor_diff(data)
        gan_diff_object = GANdiffraction(data, mask, iter_num=iter_num, recon_monitor=False, 
                                         save_wpath= '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/weights/')
        start = time.time()
        abs, rec = gan_diff_object.recon
        end = time.time()
        print('Running time is {}'.format(end - start))
            # print(spath+os.path.splitext(file_name)[0][-6:]+'.tiff')
        save_tiff(rec.reshape((px, px)), spath+os.path.splitext(file_name)[0][-6:]+'.tiff')
        
def main():

    iter_num = 2000
    # fname_data = '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/data_crop_nor.tiff'
    fname_data = '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/crop_bin/796720'
    fname_mask = '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/mask_crop_nor.tiff'
    data = dxchange.read_tiff(fname_data)
    # data  = data[128-100:128+100, 128-100:128+100]
    mask = dxchange.read_tiff(fname_mask)
    px, _ = data.shape
    data = nor_diff(data)
    gan_diff_object = GANdiffraction(data, mask, iter_num=iter_num)
    start = time.time()
    abs, rec = gan_diff_object.recon
    end = time.time()
    print('Running time is {}'.format(end - start))
    print(rec.shape, rec.dtype)
    dxchange.write_tiff(rec.reshape((px, px)), '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/crop/results/phase_log50', overwrite=True)
    dxchange.write_tiff(abs.reshape((px, px)), '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/RLi_sbcc_saxs/crop/results/abs_log50', overwrite=True)
    # dxchange.write_tiff(rec.reshape((px, px)), '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/results/phase_loge3', overwrite=True)
    # dxchange.write_tiff(abs.reshape((px, px)), '/nsls2/data/staff/xyang4/data/diffraction_ruipeng/results/abs_loge3', overwrite=True)