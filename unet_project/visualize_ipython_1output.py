#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:29:02 2019

@author: DD

visualize images, masks and predictions and saves them

this script is for a model with two outputs and only one category or channel
as output.

Right now this script should only work for images that are read in as uint8
by the imread function in this script.
"""
import functools
import os
import platform
import random

import numpy as np
import skimage
from skimage.io import imread, imsave
from skimage.color import gray2rgb, rgba2rgb
from skimage.transform import resize
import tensorflow as tf
from tensorflow.python.keras import models

from modules.library import get_img_paths
from modules.my_pyplot import display_image_in_actual_size
from modules.unet_vgg16 import DivByRedSum, vgg16_enc_dec_model

#### START GLOBALS ####
path_train = os.getcwd() + "/dataset/fixations/train"
path_test = os.getcwd() + "/dataset/fixations/test"

path_train = "/Users/jochen/Projects/antraege/bmbf-alex/material/SALICON/train"
path_test = "/Users/jochen/Projects/antraege/bmbf-alex/material/SALICON/test"



#if model path is given, visualization contains predictions
#model_path = 'save/vgg16_unet_e50_basic/trained_model.h5'
model_path = '/Users/jochen/Projects/antraege/bmbf-alex/shk/Mendel/fixations/SALICON_480_640_bce.h5'

#if no model path but weights path, weights are used. find definition in code
#to change parameters of model.
#weights_path = 'save/vgg16_unet_dbl_out/distribution_model_dbl_out_scaled_e100_callbacks_weights.hdf5'

#choose train or validation dataset settings 'tr' 'val'
tr_or_val = 'val'
#tr_or_val = 'tr'

seed = 456 #TODO: If real randomization is wanted: np.random.randint(10000)
random.seed(seed)
#image dimensions here has to fit the expected image dimensions of the model
img_height = 480
img_width  = 640
img_channels = 3

#number of random shows. If more than total set, total set is shown.
nb_random_shows = 1000

scale_shows_by = 1.0

save_predictions = True
additional_save_collage = True
save_directory = 'save_visualisations'
save_suffix_pred = '_pred'
save_suffix_pred_raw = '_pred_raw'
save_suffix_collage = '_collage'
#### END GLOBALS ####

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
  delta = 0.00000001
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
  display_list_row0 = [img_rgb]
  vertical_conc_list = []

  if len(mask.shape) == 3:
    display_list_row0.append(overlay(img_rgb, mask[:,:,0]/np.sum(mask[:,:,0])))
  else:
    display_list_row0.append(overlay(img_rgb, mask/np.sum(mask)))

  print_list = ['Image next to overlayed mask(s)']
  vertical_conc_list.append(np.concatenate(display_list_row0, axis = 1))

  if model:
    display_list_pred = [img_rgb]
    pred_raw = model.predict(img2batch(img_rgb))
    pred = pred_raw[0,:,:,:]

    display_list_pred.append(overlay(img_rgb, pred[:,:,0]))

    print_list.append('Image next to overlayed prediction(s)')
    vertical_conc_list.append(np.concatenate(display_list_pred, axis = 1))

    print_list.append('Image next to raw prediction(s)')
    cast_pred = lambda p : np.around(gray2rgb(p*255)).astype(np.uint8)
    display_list_pred_raw = [img_rgb,
                             cast_pred(pred[:,:,0])]
    vertical_conc_list.append(np.concatenate(display_list_pred_raw, axis = 1))
  for p in print_list:
    print(p)
  collage = np.concatenate(vertical_conc_list, axis = 0)
  display_image_in_actual_size(collage)

  if display_list_pred and display_list_pred_raw:
    return (display_list_row0, display_list_pred, display_list_pred_raw),collage
  else:
    return display_list_row0, collage

#TODO: SLOPPY! THIS IS JUST A COPY OF THE DEFINITIONS IN THE TRAINING SCRIPT
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
if platform.system() == 'Darwin':
  os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
if model_path:
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
  print('Using normal model.')
  model = vgg16_enc_dec_model((img_height, img_width, img_channels),
                              freeze_vgg = False,
                              output_choice = 'sigmoid',
                              **{'reg':None,'dropout':0.1})
  model.load_weights(weights_path)

if tr_or_val == 'tr':
  image_paths = get_img_paths(path_train+"/images")
  mask_paths = get_img_paths(path_train+"/masks")
else:
  image_paths = get_img_paths(path_test+"/images")
  mask_paths = get_img_paths(path_test+"/masks")


def save(display_rows, collage, image_path):
  """save function working heavily with global variables and shadowing
  others."""
  if len(display_rows) == 1:
    return
  if save_predictions:
    image_name = os.path.basename(image_path)
    image_base, image_ext = os.path.splitext(image_name)
    dir_ = save_directory
    pred_path = os.path.join(dir_, image_base + save_suffix_pred + image_ext)
    imsave(pred_path, display_rows[1][1])
    print('Saved prediction at {}'.format(pred_path))
    pred_raw_path = os.path.join(dir_,
                                 image_base + save_suffix_pred_raw + image_ext)
    imsave(pred_raw_path, display_rows[2][1])
    print('Saved raw prediction at {}'.format(pred_raw_path))
    if additional_save_collage:
      collage_path = os.path.join(dir_,
                                  image_base + save_suffix_collage + image_ext)
      imsave(collage_path, collage)
      print('Saved collage at {}'.format(collage_path))

#TODO: If there is a problem with NaN values, one can try this:
#for n,layer in enumerate(model.layers):
#  weights = layer.get_weights()
#  new_weights = [np.nan_to_num(w) for w in weights]
#  layer.set_weights(new_weights)
#%%
if nb_random_shows > len(image_paths):
  print('Showing all images since number of random shows exceed set lengths.')
  for n, (path,mask) in enumerate(zip(image_paths, mask_paths)):
    print(n,path)
    img = imread(path)
    img = np.uint8(resize(img, (img_height, img_width, 3), preserve_range=True))
    mask = imread(mask)
    mask = np.uint8(resize(mask, (img_height, img_width), preserve_range=True))
    display_rows, collage = visualize(img, mask, model)
    save(display_rows, collage, path)
else:
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
    display_rows, collage = visualize(img, mask, model)
    save(display_rows, collage, path)
