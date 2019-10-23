from __future__ import print_function
import keras
import tensorflow as tf
import numpy as np
import re, math
import os, glob, sys, threading, random
import scipy.io
from scipy import ndimage, misc
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


#None type in image dimension helps to process training with any patch dimensions. Network Can handle multiple scales.
IMG_SIZE = (None, None, 1)
BATCH_SIZE =8
EPOCHS = 200
TRAIN_SCALES = [2,3,4]
VALID_SCALES = [4]


#Function to Generate Matlab File List
def get_image_list(data_path, scales=[2, 3, 4]):
    
	l = glob.glob(os.path.join(data_path,"*"))
	l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
	train_list = []
	for f in l:
		if os.path.exists(f):	
			for i in range(len(scales)):
				scale = scales[i]
				string_scale = "_" + str(scale) + ".mat"
				if os.path.exists(f[:-4]+string_scale): 
					train_list.append([f, f[:-4]+string_scale])
	return train_list

##Function to Generate Image Batch List
def get_image_batch(train_list, offset):
    
	target_list = train_list[offset:offset+BATCH_SIZE]
	input_list = []
	gt_list = []
	for pair in target_list:
		input_img = scipy.io.loadmat(pair[1])['patch']
		gt_img = scipy.io.loadmat(pair[0])['patch']
		input_list.append(input_img)
		gt_list.append(gt_img)
	input_list = np.array(input_list)
	input_list.resize([BATCH_SIZE, np.size(input_img, 0),np.size(input_img, 1), IMG_SIZE[2]])
	gt_list = np.array(gt_list)
	gt_list.resize([BATCH_SIZE, np.size(gt_img, 0), np.size(gt_img, 1), IMG_SIZE[2]])
	return input_list, gt_list


#Function to Generate Image Batch
def image_gen(target_list):
    
	while True:
		for step in range(len(target_list)//BATCH_SIZE):
			offset = step*BATCH_SIZE
			batch_x, batch_y = get_image_batch(target_list, offset)
			yield (batch_x, [batch_y, batch_y, batch_y])


def PSNR(y_true, y_pred):
    
	return K.expand_dims(tf.image.psnr(y_true, y_pred, max_val=1.0),0)

def SSIM(y_true, y_pred):
    
	return K.expand_dims(tf.image.ssim(y_true, y_pred, max_val=1.0),0)

def step_decay(epoch):
    
	initial_lrate = 10**(-4)
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

#Residual Block
def RES_Block(x, res_number = 1):
    
	model_input = Input(shape=(None, None, 64))
	model_conv_1 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(model_input)
	model_conv_2 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(model_conv_1)
	model_conv_3 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(model_conv_2)
    	model_conv_4 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(model_conv_3)   
	model = add([model_input, model_conv_4])
	model_add = Model(model_input, model)
	for res in range(res_number):
		x = model_add(x)
	return x

