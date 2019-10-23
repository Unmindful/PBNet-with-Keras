from __future__ import print_function
import keras
import tensorflow as tf
import numpy as np
import re
import math
import os, glob, sys, threading
import scipy.io
from scipy import ndimage, misc
from keras import backend as K
from keras import optimizers, regularizers
from keras.losses import mean_squared_error, mean_absolute_error 
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense, Activation, BatchNormalization
from keras.layers import SeparableConv2D, MaxPooling2D, Input, ZeroPadding2D, merge, add, Conv2D, concatenate, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from sr_utilities import IMG_SIZE, BATCH_SIZE, EPOCHS, TRAIN_SCALE, VALID_SCALE, get_image_list, image_gen, PSNR, SSIM, step_decay, RES_Block


#Get the training and testing data
train_list = get_image_list("./Train Data/train_291_128x128bicY/", scales=TRAIN_SCALES)
test_list = get_image_list("./Validation Data/val_Set14_128x128bicY/", scales=VALID_SCALES)

input_img = Input(shape=IMG_SIZE)

#Input layer/Feature Extraction layer
model3_one = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
model5_one = Conv2D(64, (5, 5), padding='same', kernel_initializer='he_normal')(input_img)
model7_one = Conv2D(64, (7, 7), padding='same', kernel_initializer='he_normal')(input_img)
model9_one = Conv2D(64, (9, 9), padding='same', kernel_initializer='he_normal')(input_img)
model11_one = Conv2D(64, (11, 11), padding='same', kernel_initializer='he_normal')(input_img)
model13_one = Conv2D(64, (13, 13), padding='same', kernel_initializer='he_normal')(input_img)
feature_in = concatenate([model3_one, model5_one, model7_one, model9_one, model11_one, model13_one])
model = Activation('relu')(feature_in)

#1st Parallel Block
model3 = Conv2D(64, (1, 3), padding='same', kernel_initializer='he_normal')(model)
model3 = Conv2D(64, (3, 1), padding='same', kernel_initializer='he_normal')(model3)
model3_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model3)

model5 = Conv2D(64, (1, 5), padding='same', kernel_initializer='he_normal')(model)
model5 = Conv2D(64, (5, 1), padding='same', kernel_initializer='he_normal')(model5)
model5_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model5)

model7 = Conv2D(64, (1, 7), padding='same', kernel_initializer='he_normal')(model)
model7 = Conv2D(64, (7, 1), padding='same', kernel_initializer='he_normal')(model7)
model7_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model7)

model9 = Conv2D(64, (1, 9), padding='same', kernel_initializer='he_normal')(model)
model9 = Conv2D(64, (9, 1), padding='same', kernel_initializer='he_normal')(model9)
model9_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model9)

model11 = Conv2D(64, (1, 11), padding='same', kernel_initializer='he_normal')(model)
model11 = Conv2D(64, (11, 1), padding='same', kernel_initializer='he_normal')(model11)
model11_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model11)

model13 = Conv2D(64, (1, 13), padding='same', kernel_initializer='he_normal')(model)
model13 = Conv2D(64, (13, 1), padding='same', kernel_initializer='he_normal')(model13)
model13_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model13)

model3_con= concatenate([model, model3, model3_one, model5_one, model7_one, model9_one, model11_one, model13_one])
model3 = Activation('relu')(model3_con)
model5_con= concatenate([model, model5, model3_one, model5_one, model7_one, model9_one, model11_one, model13_one])
model5 = Activation('relu')(model5_con)
model7_con= concatenate([model, model7, model3_one, model5_one, model7_one, model9_one, model11_one, model13_one])
model7 = Activation('relu')(model7_con)
model9_con= concatenate([model, model9, model3_one, model5_one, model7_one, model9_one, model11_one, model13_one])
model9 = Activation('relu')(model9_con)
model11_con= concatenate([model, model11, model3_one, model5_one, model7_one, model9_one, model11_one, model13_one])
model11 = Activation('relu')(model11_con)
model13_con = concatenate([model, model13, model3_one, model5_one, model7_one, model9_one, model11_one, model13_one])
model13 = Activation('relu')(model13_con)

#2nd Parallel Block
model3 = Conv2D(64, (1, 3), padding='same', kernel_initializer='he_normal')(model3)
model3 = Conv2D(64, (3, 1), padding='same', kernel_initializer='he_normal')(model3)
model3_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model3)

model5 = Conv2D(64, (1, 5), padding='same', kernel_initializer='he_normal')(model5)
model5 = Conv2D(64, (5, 1), padding='same', kernel_initializer='he_normal')(model5)
model5_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model5)

model7 = Conv2D(64, (1, 7), padding='same', kernel_initializer='he_normal')(model7)
model7 = Conv2D(64, (7, 1), padding='same', kernel_initializer='he_normal')(model7)
model7_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model7)

model9 = Conv2D(64, (1, 9), padding='same', kernel_initializer='he_normal')(model9)
model9 = Conv2D(64, (9, 1), padding='same', kernel_initializer='he_normal')(model9)
model9_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model9)

model11 = Conv2D(64, (1, 11), padding='same', kernel_initializer='he_normal')(model11)
model11 = Conv2D(64, (11, 1), padding='same', kernel_initializer='he_normal')(model11)
model11_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model11)

model13 = Conv2D(64, (1, 13), padding='same', kernel_initializer='he_normal')(model13)
model13 = Conv2D(64, (13, 1), padding='same', kernel_initializer='he_normal')(model13)
model13_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model13)

model3_con= concatenate([model3_con, model3, model3_one, model5_one, model7_one, model9_one, model11_one, model13_one])
model3 = Activation('relu')(model3_con)
model5_con= concatenate([model5_con, model5, model3_one, model5_one, model7_one, model9_one, model11_one, model13_one])
model5 = Activation('relu')(model5_con)
model7_con= concatenate([model7_con, model7, model3_one, model5_one, model7_one, model9_one, model11_one, model13_one])
model7 = Activation('relu')(model7_con)
model9_con= concatenate([model9_con, model9, model3_one, model5_one, model7_one, model9_one, model11_one, model13_one])
model9 = Activation('relu')(model9_con)
model11_con= concatenate([model5_con, model11, model3_one, model5_one, model7_one, model9_one, model11_one, model13_one])
model11 = Activation('relu')(model11_con)
model13_con = concatenate([model5_con, model13, model3_one, model5_one, model7_one, model9_one, model11_one, model13_one])
model13 = Activation('relu')(model13_con)

#3rd Parallel Block
model3 = Conv2D(64, (1, 3), padding='same', kernel_initializer='he_normal')(model3)
model3 = Conv2D(64, (3, 1), padding='same', kernel_initializer='he_normal')(model3)
model3_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model3)

model5 = Conv2D(64, (1, 5), padding='same', kernel_initializer='he_normal')(model5)
model5 = Conv2D(64, (5, 1), padding='same', kernel_initializer='he_normal')(model5)
model5_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model5)

model7 = Conv2D(64, (1, 7), padding='same', kernel_initializer='he_normal')(model7)
model7 = Conv2D(64, (7, 1), padding='same', kernel_initializer='he_normal')(model7)
model7_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model7)

model9 = Conv2D(64, (1, 9), padding='same', kernel_initializer='he_normal')(model9)
model9 = Conv2D(64, (9, 1), padding='same', kernel_initializer='he_normal')(model9)
model9_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model9)

model11 = Conv2D(64, (1, 11), padding='same', kernel_initializer='he_normal')(model11)
model11 = Conv2D(64, (11, 1), padding='same', kernel_initializer='he_normal')(model11)
model11_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model11)

model13 = Conv2D(64, (1, 13), padding='same', kernel_initializer='he_normal')(model13)
model13 = Conv2D(64, (13, 1), padding='same', kernel_initializer='he_normal')(model13)
model13_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model13)

model3_con= concatenate([model3_con, model3, model3_one, model5_one, model7_one, model9_one, model11_one, model13_one])
model3 = Activation('relu')(model3_con)
model5_con= concatenate([model5_con, model5, model3_one, model5_one, model7_one, model9_one, model11_one, model13_one])
model5 = Activation('relu')(model5_con)
model7_con= concatenate([model7_con, model7, model3_one, model5_one, model7_one, model9_one, model11_one, model13_one])
model7 = Activation('relu')(model7_con)
model9_con= concatenate([model9_con, model9, model3_one, model5_one, model7_one, model9_one, model11_one, model13_one])
model9 = Activation('relu')(model9_con)
model11_con= concatenate([model5_con, model11, model3_one, model5_one, model7_one, model9_one, model11_one, model13_one])
model11 = Activation('relu')(model11_con)
model13_con = concatenate([model5_con, model13, model3_one, model5_one, model7_one, model9_one, model11_one, model13_one])
model13 = Activation('relu')(model13_con)

#Bottleneck Layer, performs the squeeze of the features
model3 = Conv2D(64, (1, 3), padding='same', kernel_initializer='he_normal')(model3)
model3 = Conv2D(64, (3, 1), padding='same', kernel_initializer='he_normal')(model3)
model3_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model3)

model5 = Conv2D(64, (1, 5), padding='same', kernel_initializer='he_normal')(model5)
model5 = Conv2D(64, (5, 1), padding='same', kernel_initializer='he_normal')(model5)
model5_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model5)

model7 = Conv2D(64, (1, 7), padding='same', kernel_initializer='he_normal')(model7)
model7 = Conv2D(64, (7, 1), padding='same', kernel_initializer='he_normal')(model7)
model7_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model7)

model9 = Conv2D(64, (1, 9), padding='same', kernel_initializer='he_normal')(model9)
model9 = Conv2D(64, (9, 1), padding='same', kernel_initializer='he_normal')(model9)
model9_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model9)

model11 = Conv2D(64, (1, 11), padding='same', kernel_initializer='he_normal')(model11)
model11 = Conv2D(64, (11, 1), padding='same', kernel_initializer='he_normal')(model11)
model11_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model11)

model13 = Conv2D(64, (1, 13), padding='same', kernel_initializer='he_normal')(model13)
model13 = Conv2D(64, (13, 1), padding='same', kernel_initializer='he_normal')(model13)
model13_one = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model13)

model = concatenate([model3_one, model5_one, model7_one, model9_one, model11_one, model13_one])

#Apply three Residual Blocks
for blocks in range (3):
	model = RES_Block(model)

#Reconstruction Block
model = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(model)
model = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(model)
res_img = model

#Skip Connection
output_img = add([res_img, input_img])

model = Model(input_img, output_img)
#model.load_weights('./Saved Models/sr_PBNet_model_with200epoch')

adam = Adam(lr=0.001, decay=1e-4)
sgd = SGD(lr=1e-5, momentum=0.9, decay=1e-4, nesterov=False)
#custom_loss = mae_mssim_loss(alpha=0.8) 
model.compile(adam, loss='mae', metrics=[PSNR,SSIM])
model.summary()
filepath="./saved_weights/weights-improvement-{epoch:02d}-{val_PSNR:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_PSNR', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
lrate = LearningRateScheduler(step_decay)
callbacks_list = [checkpoint,lrate]

print("Started training")
history = model.fit_generator(image_gen(train_list), steps_per_epoch=len(train_list) // BATCH_SIZE,  \
					validation_data=image_gen(test_list), validation_steps=len(test_list) // BATCH_SIZE,
					epochs=EPOCHS, workers=32, callbacks=callbacks_list, verbose=1)

print("Done training!!!")
print("Saving the final model ...")
model.save('./Saved Models/sr_PBNet_model_with200epoch.h5')  # creates a H5 file 
del model  # deletes the existing model
