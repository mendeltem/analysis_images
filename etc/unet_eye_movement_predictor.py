#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:06:17 2019

eye movement predictor, load the model and images from the eye movement 
experiment and predict the eye movement with kernel density estimation mask. 

@author: mut,dd
"""
import os
#set the working directory
ML_PATH ="/home/mendel/Documents/work/mendel_exchange"
os.chdir(ML_PATH)
#from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow import keras
ModelCheckpoint = keras.callbacks.ModelCheckpoint
from modules.model_zoo_alpha import models_dict
from modules.library import plot_training_history
from m_model import keras_cmodel

import subprocess
from modules.model_zoo import bce_loss, dice_loss, bce_dice_loss
from modules.model_zoo import precision, recall

import tensorflow as tf

#monitor
my_dpi = 90
#path length for ploting kde images
pad_length = 27
#widh and height
#w = 1024+pad_length
#h = 768+pad_length

#saving the model path
SAVE_PATH = 'figures/memory_amodel_weights.hdf5'
SAVE_PATH_bestmodel = 'figures/memory_amodel_weights_best.hdf5'
#one experiment type path 
TRAIN_COLOR_ORIGINAL_MEMORY      = os.getcwd()+"/dataset/color_memory/original/train"
TEST_COLOR_ORIGINAL_MEMORY       = os.getcwd()+"/dataset/color_memory/original/test"
#one experiment type path 
TRAIN_COLOR_PER_HOCH_MEMORY      = os.getcwd()+"/dataset/color_memory/periphererHochpass/train"
TEST_COLOR_PER_HOCH_MEMORY       = os.getcwd()+"/dataset/color_memory/periphererHochpass/test"
#one experiment type path 
TRAIN_COLOR_PER_TIEF_MEMORY      = os.getcwd()+"/dataset/color_memory/periphererTiefpass/train"
TEST_COLOR_PER_TIEF_MEMORY       = os.getcwd()+"/dataset/color_memory/periphererTiefpass/test"
#one experiment type path 
TRAIN_COLOR_ZEN_HOCH_MEMORY      = os.getcwd()+"/dataset/color_memory/zentralerHochpass/train"
TEST_COLOR_ZEN_HOCH_MEMORY       = os.getcwd()+"/dataset/color_memory/zentralerHochpass/test"
#one experiment type path 
TRAIN_COLOR_ZEN_TIEF_MEMORY      = os.getcwd()+"/dataset/color_memory/zentralerTiefpass/train"
TEST_COLOR_ZEN_TIEF_MEMORY       = os.getcwd()+"/dataset/color_memory/zentralerTiefpass/test"


#one experiment type path 
TRAIN_COLOR_ORIGINAL_SEARCH       = os.getcwd()+"/dataset/grayscale_memory/original/train"
TEST_COLOR_ORIGINAL_SEARCH        = os.getcwd()+"/dataset/grayscale_memory/original/test"
#one experiment type path 
TRAIN_COLOR_PER_HOCH_SEARCH       = os.getcwd()+"/dataset/grayscale_memory/periphererHochpass/train"
TEST_COLOR_PER_HOCH_SEARCH        = os.getcwd()+"/dataset/grayscale_memory/periphererHochpass/test"
#one experiment type path 
TRAIN_COLOR_PER_TIEF_SEARCH       = os.getcwd()+"/dataset/grayscale_memory/periphererTiefpass/train"
TEST_COLOR_PER_TIEF_SEARCH        = os.getcwd()+"/dataset/grayscale_memory/periphererTiefpass/test"
#one experiment type path 
TRAIN_COLOR_ZEN_HOCH_SEARCH       = os.getcwd()+"/dataset/grayscale_memory/zentralerHochpass/train"
TEST_COLOR_ZEN_HOCH_SEARCH        = os.getcwd()+"/dataset/grayscale_memory/zentralerHochpass/test"
#one experiment type path 
TRAIN_COLOR_ZEN_TIEF_SEARCH       = os.getcwd()+"/dataset/grayscale_memory/zentralerTiefpass/train"
TEST_COLOR_ZEN_TIEF_SEARCH        = os.getcwd()+"/dataset/grayscale_memory/zentralerTiefpass/test"

#experiment type path
TRAIN_GRAY_MEMORY       = os.getcwd()+"/dataset/grayscale_memory/original/train"
TEST_GRAY_MEMORY        = os.getcwd()+"/dataset/grayscale_memory/original/test"
#all experiment in one folder
ALL_TRAIN_PATH = os.getcwd()+"/dataset/training"
ALL_TEST_PATH = os.getcwd()+"/dataset/validation"
#collection of 300 pictures from search
TRAIN_PATH_SEARCH   = os.getcwd()+"/dataset/both_original_perHochpass_search/train"
TEST_PATH_SEARCH   = os.getcwd()+"/dataset/both_original_perHochpass_search/test"
#collection of 300 pictures from memory
TRAIN_PATH_MEMORY   = os.getcwd()+"/dataset/both_original_perHochpass_memory/train"
TEST_PATH_MEMORY   = os.getcwd()+"/dataset/both_original_perHochpass_memory/test"
#change here

#New Data Set 
#experiment type path
BIG_DATA_TRAIN      = os.getcwd()+"/dataset/Set_01/train"
BIG_DATA_TEST        = os.getcwd()+"/dataset/Set_01/test"

PATH_TRAIN  = BIG_DATA_TRAIN
PATH_TEST   = BIG_DATA_TEST

w = 1024
h = 768
BATCH_SIZE = 32

from modules.m_preprocess import image_mask_generator,load_images_and_masks
#from modules.Image_Mask_DataGenerator import Image_DataGenerator
# Set parameters for Image base Size
IMG_WIDTH =  1024
IMG_HEIGHT = 768
IMG_CHANNELS = 3
#downsize the images and masks
down_size =  4
#Training Bachsize

#random seed
seed = 42

img_height = int(IMG_HEIGHT/down_size)
img_width  = int(IMG_WIDTH/down_size)


#load the images and masks and perform pooling before resize again
trainX,trainY = load_images_and_masks(PATH_TRAIN,
                                      img_width=img_width, 
                                      img_height=img_height)
testX,testY   = load_images_and_masks(PATH_TEST, 
                                      img_width=img_width,
                                      img_height=img_height)

#Generator
train_generator,test_generator =  image_mask_generator(trainX,
                                                       trainY,
                                                       testX,
                                                       testY,
                                                       BATCH_SIZE)


checkpointer = ModelCheckpoint(filepath=SAVE_PATH_bestmodel,
                               verbose=1, save_best_only=True)


cmodel = keras_cmodel(img_height,img_width)

#training the model
history= cmodel.fit_generator(train_generator, 
                             validation_data=test_generator, 
                             validation_steps=BATCH_SIZE/2, 
                             steps_per_epoch=len(trainX)/(BATCH_SIZE*2), 
                             callbacks=[checkpointer],
                             epochs=300).history

subprocess.call("mkdir figures/new/" , shell = True )                     
subprocess.call("mkdir figures/new/val/" , shell = True )
subprocess.call("mkdir figures/new/train/" , shell = True )


with open('figures/new/trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history, file_pi)
    
plot_training_history(history,save_dir="figures/new/") 

cmodel.load_weights(SAVE_PATH_bestmodel)

cmodel.save("models/")
#testing on test set
y = cmodel.predict(testX) 

#test plot
for im in range(len(y)):
  if im > 50: break
  plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
  plt.imshow(testX[im])
  plt.savefig("figures/new/val/"+str(im)+"_image_test_out.png") 
  plt.imshow(np.squeeze(testY[im]))
  plt.savefig("figures/new/val/"+str(im)+"_mask_a_target_out.png")
  plt.imshow(np.squeeze(y[im]))
  plt.savefig("figures/new/val/"+str(im)+"_mask_b_predict_out.png")
  
#testing on train set
y_train = cmodel.predict(trainX) 

for im in range(len(y_train)):
  if im > 93: break
  plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
  plt.imshow(trainX[im])
  plt.savefig("figures/new/train/"+str(im)+"_image_train_out.png") 
  plt.imshow(np.squeeze(trainY[im]))
  plt.savefig("figures/new/train/"+str(im)+"_mask_a_target_out.png")
  plt.imshow(np.squeeze(y_train[im]))
  plt.savefig("figures/new/train/"+str(im)+"_mask_b_predict_out.png")