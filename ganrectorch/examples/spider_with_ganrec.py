from ganrec_dataloader import * 
from utils import *   
from models import *
energy = 11
z = 7.8880960e-2 
pv = 1.04735263e-7 * 4
iter_num = 700

energy = 11
z = 7.8880960e-2
pv = 1.04735263e-7
iter_num = 700
fname_data = 'data/gan_phase/data_spider.tif'

info = {
    'path': None, #path to the folder containing the images or the image itself
    'images': io.imread(fname_data),  #you can insert the image directly as an alternative
    'idx': 2, #list(np.arange(0, 70, 10)), #is the index of the image to be used if there are multiple images in images or path
    'energy_kev': energy,  #energy of the beam
    'detector_pixel_size': pv,  #pixel size of the detector 
    'distance_sample_detector': z, #distance between the sample and the detector
    'pad': 2,            #padding for the images, usually 2 or 4, and helps to reduce the boundary effect
    'alpha': 1e-8,       #multiplies the regularization term
    'iter_num': 100,     #number of iterations
    'init_model': False, #if True, load the model from the model_path
    'output_num': 2,   #number of output images from the generator
    'transform_type': 'normalize', #can be normalize, brightness, contrast, gamma, log, sigmoid, norm
    'transform_factor': 0.5, #if brightness, contrast, gamma, sigmoid, norm is chosen, this is the factor
    'file_type': 'tif',  #can be tif or npy, tiff, 
    'device': 'cuda:3',  #can be 'cpu' or 'cuda:x'
    'abs_ratio': 0.05,   #a factor to multiply the generated absorption
    'mode' : 'constant', #can be reflect, constant, circular
    'value': 'mean',     #used when constant is chosen. Either a number or 'mean'
}

args = get_args()                           # get the arguments from the info dictionary, this can be a baseline to start with
args.update(info)                           # update the arguments with the info dictionary
dataloader = Ganrec_Dataloader(**args)      #create the dataloader 
fig = dataloader.visualize(tranformed=True, show_or_plot='show')  #visualize the data

#******************************* TRAINING  ************************************

model = make_ganrec_model(**dataloader.__dict__)
gen_loss_list, dis_loss_list, propagated_intensity_list, phase_list, attenuation_list = model.train(iter_num=iter_num, save_model=True, save_model_path='model.pth')
fig = visualize([tensor_to_np(dataloader.transformed_images), tensor_to_np(propagated_intensity_list[-1]), tensor_to_np(phase_list[-1]), tensor_to_np(attenuation_list[-1])])
model.live_plot(5) #show a video updating every 'rate' iterations