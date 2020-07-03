#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:52:26 2019

@author: Temuulen, DD
"""
from tensorflow import keras
Model = keras.models.Model
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
array_to_img = keras.preprocessing.image.array_to_img
img_to_array = keras.preprocessing.image.img_to_array
load_img = keras.preprocessing.image.load_img
to_categorical = keras.utils.to_categorical


preprocess_input = keras.applications.inception_v3.preprocess_input
#from tensorflow.python.keras.models import Model

from modules.model_zoo_alpha import models_dict
from sklearn.model_selection import train_test_split
from modules.library import get_img_paths
from skimage import io
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

import tensorflow as tf



#to check if kerras use gou
from keras import backend as K
K.tensorflow_backend._get_available_gpus()


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())




  
img_shape = (768,1024,3)

epochs= 1

inc_v3_enc_dec_model = models_dict['inc_v3'](img_shape, False, None)


inc_v3_enc_dec_model.summary()

inc_v3_enc_dec_model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



seed = 701
val_ratio = 0.15   
#example
train_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


#train_generator = train_datagen.flow_from_directory(
#        dir_training_images,  # this is the target directory
#        target_size=(768, 1024),  # all images will be resized to 150x150
#        save_to_dir = "dataset/color_memory/original/generated"
#        ) 

#
#i=0
#

dir_images  = os.getcwd()+"/dataset/color_memory/original/images/train"
dir_masks  = os.getcwd()+"/dataset/color_memory/original/masks/train"

image_paths = get_img_paths(dir_images)
label_np_imgs_paths = get_img_paths(dir_masks)


input_images  = [cv2.imread(path)  for path in image_paths ]
mask_images  = [cv2.imread(path, cv2.IMREAD_GRAYSCALE)  for path in label_np_imgs_paths ]




reshaped_input_pictures  =  [ np.array(picture.reshape((-1,)+
                         picture.shape )) for picture in input_images]
reshaped_mask_images  =  [np.array(mask.reshape(
                         mask.shape   + (1,))  ) for mask in mask_images]


reshaped_input_pictures = np.array([picture/255 for picture in input_images]).reshape(-1,768,1024,3)
reshaped_mask_images    = np.array([mask for mask in mask_images]).reshape(-1,768,1024,1)




print(reshaped_input_pictures[0].shape)
print(reshaped_mask_images[0].shape)
##


x_train, x_test, y_train, y_test = \
                    train_test_split(
                                     reshaped_input_pictures,
                                     reshaped_mask_images,
                                     test_size = val_ratio,
                                     random_state = seed)
                    
 
x_train[0].shape
mask_images[0].shape

# we create two instances with the same arguments
data_gen_args = dict( preprocessing_function = preprocess_input,
                     height_shift_range=0.1,
                     rescale=1./255,
                     zoom_range=0.2,
                      rotation_range=90,
                     width_shift_range=0.1
                     )
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)


# Provide the same seed and keyword arguments to the fit and flow methods

image_datagen.fit(reshaped_input_pictures[0], augment=True, seed=seed)
mask_datagen.fit(reshaped_mask_images[0], augment=True, seed=seed)

image_generator = image_datagen.flow(
    reshaped_input_pictures,
    class_mode="categorical",
    color_mode="rgb",
    batch_size=1,
    seed=seed)





mask_generator = mask_datagen.flow(
    reshaped_mask_images,
    class_mode="categorical",
    color_mode="grayscale",
    batch_size=1,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(mask_generator, mask_generator)



inc_v3_enc_dec_model.fit_generator(
    train_generator,
    steps_per_epoch=20,
    epochs=1)



inc_v3_enc_dec_model.fit(   
    x_train,
    y_train,
    validation_split=0.1,
    steps_per_epoch=20,
    epochs=1)
 




inc_v3_enc_dec_model.inputs
inc_v3_enc_dec_model.outputs
#
#
#x_train[0].shape




#
#
#
#
#
#
#
#
#
##train_datagen.fit(x_train)
#
## we create two instances with the same arguments
#data_gen_args_image = dict(featurewise_center=True,
#                     featurewise_std_normalization=True,
#                     rotation_range=90,
#                     width_shift_range=0.1,
#                     height_shift_range=0.1,
#                     zoom_range=0.2
#                     
#                     )
#
#
#
#data_gen_args_mask = dict(featurewise_center=True,
#                     featurewise_std_normalization=True,
#                     rotation_range=90,
#                     width_shift_range=0.1,
#                     height_shift_range=0.1,
#                     zoom_range=0.2
#                     
#                     )
#
#
#
#
#image_datagen = ImageDataGenerator(**data_gen_args_image)
#mask_datagen = ImageDataGenerator(**data_gen_args_mask)
#
#
#
## Provide the same seed and keyword arguments to the fit and flow methods
#seed = 1
#image_datagen.fit(x_train, augment=True, seed=seed)
#mask_datagen.fit(y_train, augment=True, seed=seed)
#
#
#
#
#image_generator = image_datagen.flow_from_directory(
#    dir_images,
#    class_mode=None,
#    target_size=(768, 1024), 
#    seed=seed)
#
#mask_generator = mask_datagen.flow_from_directory(
#    dir_masks,
#    class_mode=None,
#    target_size=(768, 1024), 
#    seed=seed)
#
#
#
#
## combine generators into one which yields image and masks
#train_generator = zip(image_generator, mask_generator)
#
#
#batch_size =1
#
#
#inc_v3_enc_dec_model.fit_generator(
#    image_generator,
#    steps_per_epoch=20,
#    epochs=2)
#
#
#
#
#
#
#
#
#i=0
#
##for batch in train_generator:
##  i +=1
##  if(i > 10):
##    break
##
#
#
#
#
#

