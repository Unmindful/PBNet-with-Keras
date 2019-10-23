from __future__ import print_function
import keras
import tensorflow as tf
import numpy as np
import re, math
import os, glob, sys, threading, random
import scipy.io
import time
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage, misc
from scipy.misc import imsave, imread, imresize, toimage, imshow
from skimage import filters, feature
from keras import backend as K
from keras import optimizers, regularizers
from keras.losses import mean_squared_error, mean_absolute_error 
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense, Activation, BatchNormalization
from keras.layers import SeparableConv2D, MaxPooling2D, Input, ZeroPadding2D, merge, add, Conv2D, concatenate, Dropout, Lambda, Conv2DTranspose, multiply
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow import image
from keras.preprocessing import image
from sr_utilities import IMG_SIZE, PSNR, SSIM, step_decay

#Get the testing data

data_path = './Validation Data/Test Images/Urban100_bic_rgb/'
path = './Predicted_image/Urban100_PBNet_4x'
os.mkdir(path)
BATCH_SIZE = 1
TRAIN_SCALES = [2, 3, 4]
VALID_SCALES = [4]
#Load the trained model
model = load_model('./Saved Models/sr_PBNet_model_with200epoch.h5', custom_objects={'PSNR':PSNR, 'SSIM':SSIM})
count = 0
time_c = 0

#None type object can handle any input shape
#BY means of for loop, image dimension changes dynamically

for im_name in os.listdir(data_path):
	img_raw = image.load_img(data_path+im_name)
    	img_ycbcr = img_raw.convert('YCbCr')
    	img, cb, cr = img_ycbcr.split()
    	row=np.size(img, 0)
    	col=np.size(img, 1)

#Initialize final image
    	img_final = np.zeros((row, col, 3), dtype = 'float32')

#Actual Testing begins here
	x = image.img_to_array(img)
	x = x.astype('float32') / 255
	x = np.expand_dims(x, axis=0)
	start_t = time.time()
	pred = model.predict(x)
	end_t = time.time()
	print ("end_t:",end_t,"start_t:",start_t)
	time_c=time_c+end_t-start_t
	print ("Time Consumption:",end_t-start_t)
	test_img = np.reshape(pred, (row, col))

    #Reshaping CbCr channels
	cb = image.img_to_array(cb)
	cb = np.reshape(cb, (row, col))
    	cb = cb.astype('float32') / 255
    	cr = image.img_to_array(cr)
	cr = np.reshape(cr, (row, col))
    	cr = cr.astype('float32') / 255
    
    	img_final[:,:,0] = test_img
    	img_final[:,:,1] = cr
    	img_final[:,:,2] = cb
	img_final = cv2.cvtColor(img_final, cv2.COLOR_YCrCb2RGB)

    	imsave(path+'/im'+str(count)+'.png', img_final)
	count += 1
	print ("Image Number:",count)
print ("Total Time:",time_c)
