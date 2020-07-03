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
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow import keras
ModelCheckpoint = keras.callbacks.ModelCheckpoint
from modules.model_zoo_alpha import models_dict
from modules.library import plot_training_history, plot_history


from m_model import keras_model, keras_cmodel

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
w = 1024
h = 768

from modules.m_preprocess import image_mask_generator,load_images_and_masks
#from modules.Image_Mask_DataGenerator import Image_DataGenerator
# Set parameters for Image base Size
#original Imagesize
IMG_WIDTH =  1024
IMG_HEIGHT = 768
IMG_CHANNELS = 3
#downsize the images and masks
down_size =  4
height = int(IMG_HEIGHT / down_size)
width  = int(IMG_WIDTH/ down_size)
#Training Bachsize


directoryname = "TRAIN_SOL/"

#random seed
seed = 42
random.seed = seed
np.random.seed(seed=seed)
#saving the model path
SAVE_PATH = 'figures/'+str(directoryname)+'memory_amodel_weights.hdf5'
SAVE_PATH_bestmodel = 'figures/memory_amodel_weights_best.hdf5'
#one experiment type path 
TRAIN_COLOR_ORIGINAL_MEMORY = os.getcwd()+"/dataset/color_memory/original/train"
TEST_COLOR_ORIGINAL_MEMORY  = os.getcwd()+"/dataset/color_memory/original/test"
#one experiment type path 
TRAIN_COLOR_PER_HOCH_MEMORY = os.getcwd()+"/dataset/color_memory/periphererHochpass/train"
TEST_COLOR_PER_HOCH_MEMORY  = os.getcwd()+"/dataset/color_memory/periphererHochpass/test"
#one experiment type path 
TRAIN_COLOR_PER_TIEF_MEMORY = os.getcwd()+"/dataset/color_memory/periphererTiefpass/train"
TEST_COLOR_PER_TIEF_MEMORY  = os.getcwd()+"/dataset/color_memory/periphererTiefpass/test"
#one experiment type path 
TRAIN_COLOR_ZEN_HOCH_MEMORY = os.getcwd()+"/dataset/color_memory/zentralerHochpass/train"
TEST_COLOR_ZEN_HOCH_MEMORY  = os.getcwd()+"/dataset/color_memory/zentralerHochpass/test"
#one experiment type path 
TRAIN_COLOR_ZEN_TIEF_MEMORY = os.getcwd()+"/dataset/color_memory/zentralerTiefpass/train"
TEST_COLOR_ZEN_TIEF_MEMORY  = os.getcwd()+"/dataset/color_memory/zentralerTiefpass/test"

#one experiment type path 
TRAIN_COLOR_ORIGINAL_SEARCH = os.getcwd()+"/dataset/color_search/original/train"
TEST_COLOR_ORIGINAL_SEARCH  = os.getcwd()+"/dataset/color_search/original/test"
#one experiment type path 
TRAIN_COLOR_PER_HOCH_SEARCH = os.getcwd()+"/dataset/color_search/periphererHochpass/train"
TEST_COLOR_PER_HOCH_SEARCH  = os.getcwd()+"/dataset/color_search/periphererHochpass/test"
#one experiment type path 
TRAIN_COLOR_PER_TIEF_SEARCH = os.getcwd()+"/dataset/color_search/periphererTiefpass/train"
TEST_COLOR_PER_TIEF_SEARCH  = os.getcwd()+"/dataset/color_search/periphererTiefpass/test"
#one experiment type path 
TRAIN_COLOR_ZEN_HOCH_SEARCH = os.getcwd()+"/dataset/color_search/zentralerHochpass/train"
TEST_COLOR_ZEN_HOCH_SEARCH  = os.getcwd()+"/dataset/color_search/zentralerHochpass/test"
#one experiment type path 
TRAIN_COLOR_ZEN_TIEF_SEARCH = os.getcwd()+"/dataset/color_search/zentralerTiefpass/train"
TEST_COLOR_ZEN_TIEF_SEARCH  = os.getcwd()+"/dataset/color_search/zentralerTiefpass/test"


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

ALL_TRAIN_SEARCH_MEMORY  = os.getcwd()+"/dataset/memory_search/train"
ALL_TEST_SEARCH_MEMORY  = os.getcwd()+"/dataset/memory_search/test"

#New Data Set 
#experiment type path
BIG_DATA_TRAIN      = os.getcwd()+"/dataset/Set_01/train"
BIG_DATA_TEST        = os.getcwd()+"/dataset/Set_01/test"

PATH_TRAIN  = BIG_DATA_TRAIN
PATH_TEST   = BIG_DATA_TEST

#checkpoint
checkpointer = ModelCheckpoint(filepath=SAVE_PATH_bestmodel,
                               verbose=1, save_best_only=True)



from generator_for_mask import ImageSequence



mean_tr_acc = []
mean_tr_loss = []
mean_val_acc = []
mean_val_loss = []
model = keras_cmodel(height,width)






seq =  ImageSequence(BIG_DATA_TRAIN,batch_size=64, target_size= (height ,width))  

print("Epoche 1")


for i in range(seq.__len__()):
  print(i)
  shape_image = next(seq.__getitem__(i))[0].shape
  shape_mask = next(seq.__getitem__(i))[1].shape
  print("image: ", shape_image)
  print("mask: ", shape_mask)

  seq_r = next(seq.__getitem__(i))

  tr_loss, tr_acc = model.train_on_batch(seq_r[0], seq_r[1]) 
  
  
mean_tr_acc.append(tr_acc)
mean_tr_loss.append(tr_loss)
  
  
  
  
  
#for i in range(seq.__len__()):
#  print(i)
#  
#  
#  
#  
#  
#  seq_r = next(seq.__getitem__(i))
#  
#  image = seq_r[0][14]
#  #mask = seq_r[1][1]
#
#  plt.imshow(image)
#  plt.show()
#
#
#
#
#
#
#
#plt.imshow(np.squeeze(mask))
#
#
#ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
#
## Image data generator distortion options
#data_gen_args = dict(rotation_range=45.,
#                     width_shift_range=0.2,
#                     height_shift_range=0.2,
#                     shear_range=0.2,
#                     zoom_range=0.2,
#                     horizontal_flip=True,
#                     vertical_flip=True,
#                     fill_mode='reflect')  #use 'constant'??


# Train data, provide the same seed and keyword arguments to the fit and flow methods
#X_datagen = ImageDataGenerator(**data_gen_args)
#Y_datagen = ImageDataGenerator(**data_gen_args)
#
#
#image = list(seq.__getitem__(5))[0][0] / 255
#mask = list(seq.__getitem__(5))[0][1][2][:,:,2]/ 255
#
#
#
#
#
#
#[1][0] / 255
#
#
#
#
#X_datagen.fit(image, augment=True, seed=seed)
#
#
#Y_datagen.fit(trainY, augment=True, seed=seed)
#
#





#
#
#
#
#plt.imshow(image)
#
#
##training the model
#history= model.fit_generator( seq.__getitem__(BATCHSIZE), 
#                             steps_per_epoch = seq.__len__(),
#                             callbacks=[checkpointer],
#                             epochs=200)
#
#
#
#
##subprocess.call("mkdir figures/ALL_TEST_SEARCH_MEMORY","Downsize_",down_size,"/" , shell = True )
##load the images and masks and perform pooling before resize again
#trainX,trainY = load_images_and_masks(PATH_TRAIN,down_size, method = "mean",pool_size = 1)
#testX,testY   = load_images_and_masks(PATH_TEST,down_size, method = "mean",pool_size = 1)
#
#
#
##load the model
#
#
#
#BATCH_SIZE = 8
##Generator
#train_generator,test_generator =  image_mask_generator(trainX,
#                                                       trainY,
#                                                       testX,
#                                                       testY,
#                                                       BATCH_SIZE)

#loss = 'bce_dice_loss'
#metrics = ['dice_loss']
#
##dict to make variable selection simpler
#metrics_and_losses = {'dice_loss':dice_loss,
#                       'bce_dice_loss':bce_dice_loss,
#                       'bce_loss':bce_loss,
#                       'precision': precision,
#                       'recall': recall}

##image model shape
#img_shape = (int(IMG_HEIGHT / down_size), 
#             int(IMG_WIDTH/ down_size),IMG_CHANNELS)
#
#model = models_dict['vgg16'](img_shape, False, None)
##
##
##model.summary()
##
#model.compile(optimizer='adam',
#            loss=metrics_and_losses[loss],
#            metrics=[metrics_and_losses[m] for m in metrics])  

#model.summary()
#for i,layer in enumerate(model.layers):
#  print(layer,"",layer.trainable)
#  

#
#
##training the model
#history= model.fit_generator(train_generator, 
#                             validation_data=test_generator, 
#                             validation_steps=BATCH_SIZE/2, 
#                             steps_per_epoch=len(trainX)/(BATCH_SIZE*2), 
#                             callbacks=[checkpointer],
#                             epochs=200)
#
#with open("figures/"+str(directoryname)+"trainHistoryDict", 'wb') as file_pi:
#    pickle.dump(history, file_pi)
#    
#plot_training_history(history.history,save_dir="figures/"+str(directoryname)+"loss"  ) 
#
#history.history["val_dice_loss"][-1]
#    
#model.load_weights(SAVE_PATH_bestmodel)
#
##testing on test set
#y = model.predict(testX) 
#
#
##
##
#try:
#  subprocess.call("mkdir figures/"+str(directoryname) , shell = True )
#  subprocess.call("mkdir figures/"+str(directoryname)+"val/" , shell = True )
#  subprocess.call("mkdir figures/"+str(directoryname)+"train/" , shell = True )
#except:
#  pass
#
##test plot
#for im in range(len(y)):
#  if im > 30: break
#  plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
#  plt.imshow(testX[im])
#  plt.savefig("figures/"+str(directoryname)+"val/"+str(im)+"_image_test_out.png") 
#  plt.imshow(np.squeeze(testY[im]))
#  plt.savefig("figures/"+str(directoryname)+"val/"+str(im)+"_mask_a_target_out.png")
#  plt.imshow(np.squeeze(y[im]))
#  plt.savefig("figures/"+str(directoryname)+"val/"+str(im)+"_mask_b_predict_out.png")
#  
##testing on train set
#y_train = model.predict(trainX[:100]) 
#
#
#for im in range(len(y_train)):
#  if im > 93: break
#  plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
#  plt.imshow(trainX[im])
#  plt.savefig("figures/"+str(directoryname)+"train/"+str(im)+"_image_train_out.png") 
#  plt.imshow(np.squeeze(trainY[im]))
#  plt.savefig("figures/"+str(directoryname)+"train/"+str(im)+"_mask_a_target_out.png")
#  plt.imshow(np.squeeze(y_train[im]))
#  plt.savefig("figures/"+str(directoryname)+"train/"+str(im)+"_mask_b_predict_out.png")

