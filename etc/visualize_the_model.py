#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:06:17 2019

@author: mut,dd

visualize the result of the eyemovement predictor
"""
from scipy import misc
import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain
import matplotlib.image as mpimg
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from keras.models import load_model
from modules.model_zoo import bce_loss, dice_loss, bce_dice_loss
from modules.model_zoo import precision, recall


from modules.library import plot_training_history
#from b_model import keras_model

import pickle

import tensorflow as tf
from tensorflow import keras
Model = keras.models.Model
preprocess_input = keras.applications.inception_v3.preprocess_input
EarlyStopping = keras.callbacks.EarlyStopping
ModelCheckpoint = keras.callbacks.ModelCheckpoint
SGD = keras.optimizers.SGD

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#from keras.optimizers import SGD
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
activations = keras.activations
#from generator import ImageDataGenerator
from modules.model_zoo_alpha import models_dict
from modules.library import get_img_paths
from PIL import Image
#from modules.Image_Mask_DataGenerator import Image_DataGenerator
# Set some parameters
IMG_WIDTH =  1024
IMG_HEIGHT = 768
IMG_CHANNELS = 3


#one experiment type path 
TRAIN_COLOR_MEMORY      = os.getcwd()+"/dataset/color_memory/original/train"
TEST_COLOR_MEMORY       = os.getcwd()+"/dataset/color_memory/original/test"

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
PATH_TRAIN  = TRAIN_PATH_MEMORY
PATH_TEST   = TEST_PATH_MEMORY

down_size = 4
BATCH_SIZE = 8


def load_images_and_masks(path, down_size = 2, br = 100000):
  """Load the images and preprocess for the training and prediction
  
  input: path to the images and mask in two seperate directorys
  
  returns:  images, masks
  """
  all_image_paths = get_img_paths(path+ "/images/" )
  train_ids = [os.path.basename(i) for i in all_image_paths]
  
  X_images = np.zeros((len(train_ids), int(IMG_HEIGHT / down_size), int(IMG_WIDTH/ down_size), 3), dtype=np.float)
  Y_images = np.zeros((len(train_ids), int(IMG_HEIGHT /down_size), int(IMG_WIDTH/down_size), 1), dtype=np.float)
  
  i = 0
  for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    image = np.zeros((IMG_HEIGHT,IMG_WIDTH, 3), dtype=np.float)
  
    try:
      img = imread(path + "/images/"  +id_)
      
      if len(img.shape) == 3:
          image = img[:,:,:3]
      else:
          image[:,:,0] = img[:,:]
          image[:,:,1] = img[:,:]
          image[:,:,2] = img[:,:]
          
      image = resize(image, ( int(IMG_HEIGHT / down_size), int(IMG_WIDTH/ down_size)), mode='constant', preserve_range=True)
      
      
      
      mask = imread(path+ "/masks/" + id_)
      mask = np.expand_dims(resize(mask, ( int(IMG_HEIGHT / down_size), int(IMG_WIDTH/ down_size)), mode='constant', 
                                      preserve_range=True), axis=-1)
      
       
      X_images[i] = image / 255
      Y_images[i] = mask / 255
      i +=1
      if i == br: break
  
    except:
      print("ERROR")
      
  return X_images, Y_images
  
#load the test images and masks
testX,testY   = load_images_and_masks(PATH_TEST,down_size)

#load compiled model
from m_model import keras_model,keras_cmodel

#model = keras_model(int(IMG_HEIGHT/down_size),int(IMG_WIDTH/down_size))

model = keras_cmodel(int(IMG_HEIGHT/down_size),int(IMG_WIDTH/down_size))

model.load_weights("figures/memory_amodel_weights.hdf5")

model.load_weights("figures/memory_amodel_weights_best.hdf5")

history = pickle.load( open( "figures/trainHistoryDict", "rb" ) )

y = model.predict(testX) 

for im in range(len(y)):
  
    imshow(testX[im])
    #plt.show()
    plt.savefig("figures/"+str(im)+"_image_out.png") 
    imshow(np.squeeze(testY[im]))
    #plt.show()
    plt.savefig("figures/"+str(im)+"_mask_a_target_out.png")
    imshow(np.squeeze(y[im]))
    #plt.show()
    plt.savefig("figures/"+str(im)+"_mask_b_best_predict_out.png")


plot_training_history(history)


