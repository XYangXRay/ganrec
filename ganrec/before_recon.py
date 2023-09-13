# %%
import time
import numpy as np
import dxchange
from utils import nor_phase
from ganrec2 import GANphase
import os
from lib import visualize
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from ganrec1 import *
import skimage.io as io

# %%
energy = 11
z = 7.8880960e-2
pv = 1.04735263e-7
iter_num = 700

fname_data = 'data/gan_phase/data_spider.tif'
kwargs = {'energy': energy, 'z': z, 'pv': pv, 'iter_num': iter_num, 'phase_only': False, 'save_wpath': 'data/gan_phase/spider_abs_ratio/', 'init_wpath': 'data/gan_phase/spider_abs_ratio/', 'init_model': False}
data = dxchange.read_tiff(fname_data)
nprj, px,  py = data.shape
data = nor_phase(data)
abs_ratio = 0.01

# %%


# %%
abs_ratio_all = np.arange(0.01, 10*0.04, 0.004)
abs_ratio_all = 0.005*np.ones(180)

absorption = np.zeros(shape=(abs_ratio_all.shape[0], px, py), dtype=np.float32)
phase = np.zeros(shape=(abs_ratio_all.shape[0], px, py), dtype=np.float32)
propagated = np.zeros(shape=(abs_ratio_all.shape[0], px, py), dtype=np.float32)
loss = np.zeros(shape=(abs_ratio_all.shape[0], 1), dtype=np.float32)
time_count = np.zeros(shape=(abs_ratio_all.shape[0], 1), dtype=np.float32)

from ganrec_dataloader import measure_reconstruction_quality, tf_reshape, tfback_phase
kwargs['iter_num'] = 1500
kwargs['internal_iter'] = 10
kwargs['last_retrieval'] = False
kwargs['save_wpath'] = 'data/gan_phase/spider_abs_ratio/'
kwargs['init_wpath'] = 'data/gan_phase/spider_abs_ratio/'
kwargs['init_model'] = False
kwargs['save_model'] = False
kwargs['recon_monitor'] = False
kwargs['filter_type'] = 'median' #chose from median, contrast, noise, diffuse, phase_only, alternate
side_propagations = []
length = 180


abs_ratio_all = 0.005*np.ones(length)
for i, abs_ratio in enumerate(abs_ratio_all):
    idx = i
    kwargs['abs_ratio'] = abs_ratio
    gan_phase_object = GANphase(data[idx], **kwargs)
    start = time.time()
    absorption[i], phase[i],propagated[i], loss[i], side_propagation  = gan_phase_object.recon
    side_propagations.append(side_propagation)
    end = time.time()
    time_count[i] = end - start
    matched = tfback_phase( propagated[i], data[idx])
    # visualize([absorption[i], phase[i], propagated[i]])#, matched, data[idx], idx, abs_ratio, kwargs['iter_num'])
    io.imsave('/home/hailudaw/hailudaw/git_folders/ganrec/ganrec/data/gan_phase/spider_with_filters/0.005/phase/phase_'+str(idx)+'_'+str(abs_ratio)+'.tif', phase[i])
    io.imsave('/home/hailudaw/hailudaw/git_folders/ganrec/ganrec/data/gan_phase/spider_with_filters/0.005/abs/abs_'+str(idx)+'_'+str(abs_ratio)+'.tif', absorption[i])
    io.imsave('/home/hailudaw/hailudaw/git_folders/ganrec/ganrec/data/gan_phase/spider_with_filters/0.005/propagated/propagated_'+str(idx)+'_'+str(abs_ratio)+'.tif', propagated[i])
    df = measure_reconstruction_quality(img1=tf_reshape(propagated[i]), img2 = tf_reshape(data[idx]), experiment_name = 'spider_hair_'+str(idx)+'_'+str(abs_ratio), csv_file = 'Vojtech_fixed.csv', iteration = kwargs['iter_num'], save = True, epoch_time = time_count[0], total_time = np.sum(time_count))
    # display(df)

abs_ratio_all = 0.001*np.ones(length)
for i, abs_ratio in enumerate(abs_ratio_all):
    idx = i
    kwargs['abs_ratio'] = abs_ratio
    gan_phase_object = GANphase(data[idx], **kwargs)
    start = time.time()
    absorption[i], phase[i],propagated[i], loss[i], side_propagation  = gan_phase_object.recon
    side_propagations.append(side_propagation)
    end = time.time()
    time_count[i] = end - start
    matched = tfback_phase( propagated[i], data[idx])
    # visualize([absorption[i], phase[i], propagated[i]])#, matched, data[idx], idx, abs_ratio, kwargs['iter_num'])
    io.imsave('/home/hailudaw/hailudaw/git_folders/ganrec/ganrec/data/gan_phase/spider_with_filters/0.001/phase/phase_'+str(idx)+'_'+str(abs_ratio)+'.tif', phase[i])
    io.imsave('/home/hailudaw/hailudaw/git_folders/ganrec/ganrec/data/gan_phase/spider_with_filters/0.001/abs/abs_'+str(idx)+'_'+str(abs_ratio)+'.tif', absorption[i])
    io.imsave('/home/hailudaw/hailudaw/git_folders/ganrec/ganrec/data/gan_phase/spider_with_filters/0.001/propagated/propagated_'+str(idx)+'_'+str(abs_ratio)+'.tif', propagated[i])
    df = measure_reconstruction_quality(img1=tf_reshape(propagated[i]), img2 = tf_reshape(data[idx]), experiment_name = 'spider_hair_'+str(idx)+'_'+str(abs_ratio), csv_file = 'Vojtech_fixed.csv', iteration = kwargs['iter_num'], save = True, epoch_time = time_count[0], total_time = np.sum(time_count))
    # display(df)

abs_ratio_all = 0.01*np.ones(length)
for i, abs_ratio in enumerate(abs_ratio_all):
    idx = i
    kwargs['abs_ratio'] = abs_ratio
    gan_phase_object = GANphase(data[idx], **kwargs)
    start = time.time()
    absorption[i], phase[i],propagated[i], loss[i], side_propagation  = gan_phase_object.recon
    side_propagations.append(side_propagation)
    end = time.time()
    time_count[i] = end - start
    matched = tfback_phase( propagated[i], data[idx])
    # visualize([absorption[i], phase[i], propagated[i]])#, matched, data[idx], idx, abs_ratio, kwargs['iter_num'])
    io.imsave('/home/hailudaw/hailudaw/git_folders/ganrec/ganrec/data/gan_phase/spider_with_filters/0.01/phase/phase_'+str(idx)+'_'+str(abs_ratio)+'.tif', phase[i])
    io.imsave('/home/hailudaw/hailudaw/git_folders/ganrec/ganrec/data/gan_phase/spider_with_filters/0.01/abs/abs_'+str(idx)+'_'+str(abs_ratio)+'.tif', absorption[i])
    io.imsave('/home/hailudaw/hailudaw/git_folders/ganrec/ganrec/data/gan_phase/spider_with_filters/0.01/propagated/propagated_'+str(idx)+'_'+str(abs_ratio)+'.tif', propagated[i])
    df = measure_reconstruction_quality(img1=tf_reshape(propagated[i]), img2 = tf_reshape(data[idx]), experiment_name = 'spider_hair_'+str(idx)+'_'+str(abs_ratio), csv_file = 'Vojtech_fixed.csv', iteration = kwargs['iter_num'], save = True, epoch_time = time_count[0], total_time = np.sum(time_count))
    # display(df)

abs_ratio_all = 0.1*np.ones(length)
for i, abs_ratio in enumerate(abs_ratio_all):
    idx = i
    kwargs['abs_ratio'] = abs_ratio
    gan_phase_object = GANphase(data[idx], **kwargs)
    start = time.time()
    absorption[i], phase[i],propagated[i], loss[i], side_propagation  = gan_phase_object.recon
    side_propagations.append(side_propagation)
    end = time.time()
    time_count[i] = end - start
    matched = tfback_phase( propagated[i], data[idx])
    # visualize([absorption[i], phase[i], propagated[i]])#, matched, data[idx], idx, abs_ratio, kwargs['iter_num'])
    io.imsave('/home/hailudaw/hailudaw/git_folders/ganrec/ganrec/data/gan_phase/spider_with_filters/0.1/phase/phase_'+str(idx)+'_'+str(abs_ratio)+'.tif', phase[i])
    io.imsave('/home/hailudaw/hailudaw/git_folders/ganrec/ganrec/data/gan_phase/spider_with_filters/0.1/abs/abs_'+str(idx)+'_'+str(abs_ratio)+'.tif', absorption[i])
    io.imsave('/home/hailudaw/hailudaw/git_folders/ganrec/ganrec/data/gan_phase/spider_with_filters/0.1/propagated/propagated_'+str(idx)+'_'+str(abs_ratio)+'.tif', propagated[i])
    df = measure_reconstruction_quality(img1=tf_reshape(propagated[i]), img2 = tf_reshape(data[idx]), experiment_name = 'spider_hair_'+str(idx)+'_'+str(abs_ratio), csv_file = 'Vojtech_fixed.csv', iteration = kwargs['iter_num'], save = True, epoch_time = time_count[0], total_time = np.sum(time_count))
    # display(df)

abs_ratio_all = 1*np.ones(180)
for i, abs_ratio in enumerate(abs_ratio_all):
    idx = i
    kwargs['abs_ratio'] = abs_ratio
    gan_phase_object = GANphase(data[idx], **kwargs)
    start = time.time()
    absorption[i], phase[i],propagated[i], loss[i], side_propagation  = gan_phase_object.recon
    side_propagations.append(side_propagation)
    end = time.time()
    time_count[i] = end - start
    matched = tfback_phase( propagated[i], data[idx])
    # visualize([absorption[i], phase[i], propagated[i]])#, matched, data[idx], idx, abs_ratio, kwargs['iter_num'])
    io.imsave('/home/hailudaw/hailudaw/git_folders/ganrec/ganrec/data/gan_phase/spider_with_filters/1/phase/phase_'+str(idx)+'_'+str(abs_ratio)+'.tif', phase[i])
    io.imsave('/home/hailudaw/hailudaw/git_folders/ganrec/ganrec/data/gan_phase/spider_with_filters/1/abs/abs_'+str(idx)+'_'+str(abs_ratio)+'.tif', absorption[i])
    io.imsave('/home/hailudaw/hailudaw/git_folders/ganrec/ganrec/data/gan_phase/spider_with_filters/1/propagated/propagated_'+str(idx)+'_'+str(abs_ratio)+'.tif', propagated[i])
    df = measure_reconstruction_quality(img1=tf_reshape(propagated[i]), img2 = tf_reshape(data[idx]), experiment_name = 'spider_hair_'+str(idx)+'_'+str(abs_ratio), csv_file = 'Vojtech_fixed.csv', iteration = kwargs['iter_num'], save = True, epoch_time = time_count[0], total_time = np.sum(time_count))
    # display(df)

# %%
visualize(propagated)
visualize(phase)
visualize(absorption)

# %%



