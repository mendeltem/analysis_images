#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:29:02 2019

@author: DD

visualize images, masks and predictions
"""
import functools
#import json
import os
#import pickle
import platform
import random

import numpy as np
import skimage
from skimage.io import imread
from skimage.color import gray2rgb, rgba2rgb
from skimage.transform import resize
import tensorflow as tf
from tensorflow.python.keras import models

from modules.library import get_img_paths
from modules.my_pyplot import display_image_in_actual_size
from modules.unet_vgg16 import DivByRedSum

working_dir =  '/Users/dadmin/Documents/PythonStuff/Mendel'
os.chdir(working_dir)

#### START GLOBALS ####
path_train = os.getcwd() + "/dataset/Set_01/train"
path_test = os.getcwd() + "/dataset/Set_01/test"

#if model path is given, visualization contains predictions
model_path = 'save/vgg16_unet/trained_model.h5'

#choose train or validation dataset settings
tr_or_val = 'val'

#start: dataset parameters
#img_shape = (768,512,3)
seed = 706 #np.random.randint(10000)
random.seed(seed)
#batch_size = 1 #should be one
#end: dataset parameters

IMG_WIDTH =  1024
IMG_HEIGHT = 768
IMG_CHANNELS = 3
#downsize the images and masks
down_size =  2
img_height = int(IMG_HEIGHT/down_size)
img_width  = int(IMG_WIDTH/down_size)


#number of random shows
nb_random_shows = 30

scale_shows_by = 0.75
#### END GLOBALS ####

#technical parameters:
mask_area_value = 1

display_image_in_actual_size = functools.partial(display_image_in_actual_size,
                                                 scale = scale_shows_by)

def check_dtype(img, dtype = np.uint8):
  if img.dtype != dtype:
    raise Exception('Data type is not {}!'.format(dtype))


def to_rgb(img):
  """force rgb uint8 format
  """
  check_dtype(img)
  if len(img.shape) == 3:
    if img.shape[2] == 4:
      img = skimage.img_as_ubyte(rgba2rgb(img))
  else:
    img = np.stack((img,img,img), axis=2)
  return img


def overlay(img, mask):
  """overlay img with mask by displaying areas with high mask values

  Arguments:
    img: an image castable to rgb

    mask: a binary numpy array with (height, width) and a max value and zero
    as the second value.

  Returns:
    an image of type and shape of the input image with the parts visible that
    have max value on the mask.
  """
  img_type = img.dtype
  mask_f = np.float32(mask)
  img_f = np.float32(img)
  max_ = np.max(mask_f)
  delta = 0.001
  if max_ > 0 + delta:
    overlayed = np.zeros(img.shape)
    for i in range(img.shape[2]):
      overlayed[:,:,i] = img_f[:,:,i] * (mask_f/max_)
    return overlayed.astype(img_type)
  else:
    return img * 0


def img2batch(img):
  check_dtype(img)
  return np.expand_dims(img/255, axis=0)


def visualize(img, mask, model = None):
  """Visualize image with masks and predictions."""
  img_rgb = to_rgb(img)
  display_list = [img_rgb]
  vertical_conc_list = []

  if len(mask.shape) == 3:
    for i in range(mask.shape[2]):
      display_list.append(overlay(img_rgb, mask[:,:,i]))
  else:
    display_list.append(overlay(img_rgb, mask))

  print_list = ['Image next to overlayed mask(s)']
  vertical_conc_list.append(np.concatenate(display_list, axis = 1))

  if model:
    display_list_pred = [img_rgb]
    pred = model.predict(img2batch(img_rgb))[0,:,:,:]
#    print('prediction:')
#    print(np.histogram(pred))

    for i in range(pred.shape[2]):
      display_list_pred.append(overlay(img_rgb, pred[:,:,i]))
    print_list.append('Image next to overlayed prediction(s)')
    vertical_conc_list.append(np.concatenate(display_list_pred, axis = 1))

    print_list.append('Image next to raw prediction(s)')
    cast_pred = lambda p : np.around(gray2rgb(p*255)).astype(np.uint8)
    vertical_conc_list.append(np.concatenate([img_rgb] +
        [cast_pred(pred[:,:,i]) for i in range(pred.shape[2])], axis = 1))

  for p in print_list:
    print(p)
  display_image_in_actual_size(np.concatenate(vertical_conc_list, axis = 0))

#TODO: SLOPPY! THIS IS JUST A COPY OF THE DEFINITIONS IN THE TRAINING FILE
def mean_pred(y_true, y_pred):
    return tf.reduce_mean(y_pred)

def max_pred(y_true, y_pred):
    return tf.reduce_max(y_pred)

def min_pred(y_true, y_pred):
    return tf.reduce_min(y_pred)

def mean_true(y_true, y_pred):
    return tf.reduce_mean(y_true)

def max_true(y_true, y_pred):
    return tf.reduce_max(y_true)

def min_true(y_true, y_pred):
    return tf.reduce_min(y_true)

#load model
if model_path:
  if platform.system() == 'Darwin':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
  model = models.load_model(
            model_path,
            custom_objects={'DivByRedSum': DivByRedSum,
                            'mean_pred' : mean_pred,
                            'max_pred': max_pred,
                            'min_pred' : min_pred,
                            'mean_true': mean_true,
                            'max_true': max_true,
                            'min_true':min_true})
else:
  model = None

if tr_or_val == 'tr':
  image_paths = get_img_paths(path_train+"/images")
  mask_paths = get_img_paths(path_train+"/masks")
else:
  image_paths = get_img_paths(path_test+"/images")
  mask_paths = get_img_paths(path_test+"/masks")


#%%

for i in range(nb_random_shows):
  path_info = [(n,path) for n,path in enumerate(image_paths)]
  mask_info = [(n,path) for n,path in enumerate(mask_paths)]
  r = random.randint(0,len(path_info)-1)
  index = path_info[r][0]
  path = path_info[r][1]
  mask = mask_info[r][1]
  print(i, index, path)
  img = imread(path)
  img = np.uint8(resize(img, (img_height, img_width, 3), preserve_range=True))
  mask = imread(mask)
  mask = np.uint8(resize(mask, (img_height, img_width), preserve_range=True))
  vis_out = visualize(img, mask, model)
