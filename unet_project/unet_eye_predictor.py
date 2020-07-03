#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:06:17 2019

@author: mut,dd

train model to predict eye fixations

This script creates a data generator either from a saved binary or from
scratch from the given image directories. Then a loaded or newly generated
model trains on the generator and is saved at the end.

This script should be executed in a terminal. The argument parser has only
half of the important parameters. The other half has to be changed in
the script.
"""
import argparse
import os
from os.path import join, splitext
import platform
from skimage.transform import resize

import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import models
from tensorflow.python.ops import math_ops

from modules.m_preprocess import load_images_and_masks, image_mask_generator
from modules.m_preprocess import Unet_Data_Generator, plot_history, create_dir
from modules.m_preprocess import create_images
from modules.unet_vgg16 import vgg16_enc_dec_model, DivByRedSum, output_choices
from modules.mult_gpu import get_available_gpus
from modules.model_zoo import bce_loss, dice_loss, bce_dice_loss, dice_coeff
from keras.callbacks import EarlyStopping

metrics = keras.metrics
ModelCheckpoint = keras.callbacks.ModelCheckpoint
kl_divergence = keras.losses.kullback_leibler_divergence
mean_squared_error = keras.losses.mean_squared_error
binary_crossentropy = keras.losses.binary_crossentropy
multi_gpu_model = keras.utils.multi_gpu_model


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement=True



def plot_history_accuracy(history , save_dir = 0):
  for lab in history.keys():
    if "acc" in lab:
      plt.plot(history[lab], label = lab +" " + str(round(history[lab][-1], 2  )))
    if "val_acc" in lab:
      plt.plot(history[lab], label = lab +" " + str(round(history[lab][-1], 2  )))      
  plt.legend()
  if save_dir and "acc" in lab:
    plt.savefig(join(save_dir,"history_accuracy.jpg"))
  plt.close()


def plot_history_loss(history, save_dir = 0):
  for lab in history.keys():
    if "loss" == lab:
      plt.plot(history[lab], label = lab +" " + str(round(history[lab][-1], 2  )))
    if "val_loss" == lab:
      plt.plot(history[lab], label = lab +" " + str(round(history[lab][-1], 2  )))      
  plt.legend()
  if save_dir and "val_loss" in lab:
    plt.savefig(join(save_dir,"loss.jpg"))    
  plt.close()


def plot_history_bce_dice_loss(history, save_dir = 0):
  for lab in history.keys():
    if "bce_dice_loss" == lab:
      plt.plot(history[lab], label = lab +" " + str(round(history[lab][-1], 2  )))
    if "val_bce_dice_loss" == lab:
      plt.plot(history[lab], label = lab +" " + str(round(history[lab][-1], 2  )))  
      
  plt.legend()
  if save_dir and "bce_dice_loss" in lab:
    plt.savefig(join(save_dir,"bce_dice_loss.jpg"))    
  plt.close()


if platform.system() == 'Darwin':
  os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def kullback_leibler_divergence(y_true, y_pred):
  t = tf.clip_by_value(y_true, keras.backend.epsilon(), 1)
  p = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1)
  return 1e4*math_ops.reduce_sum(t * math_ops.log(t/p), axis=-1)

loss_dict = {'kl_divergence' : kl_divergence, #TODO: change loss functions if necessary
             'mean_squared_error' : mean_squared_error,
             'binary_crossentropy' : binary_crossentropy,
             'kl_and_bce_2_outputs' : [kullback_leibler_divergence,
                                       binary_crossentropy]}
             
### START SCRIPT PARAMETERS ###

#has to be chosen by parser argument
counter_overfit     = {'reg' : {'reg':0.001,'dropout':None},
                   'dropout' : {'reg':None,'dropout':0.1},
                   'none' : {'reg':None,'dropout':None}}

default_data_set            = "/mnt/data/DATA/dataset/UNET/prepared_dataset/ANKES_BW_25_control"
#data_bin_default    = 'save/trainxy_testxy_raw_192_256.p'
save_dir            = '/mnt/data/DATA/dataset/UNET/Output'
model_save_name     = 'name'

batch_size_default  = 8
epochs_default      = 2
img_base_width      = 1024
img_base_height     = 768
img_base_channels   = 3
seed                = 67
manuell_default     = 0

### END SCRIPT PARAMETERS ###

parser = argparse.ArgumentParser()

parser.add_argument('--load_model',
                    help='Path of model to load')

parser.add_argument('--save_dir',
                    help='Directory to save everything. Default is {}'.format(
                        save_dir),
                    default=save_dir)

parser.add_argument('--model_save_name',
                    help='File name of the model save file.'
                    ' Default is {}'.format(model_save_name),
                    default=model_save_name)

#parser.add_argument('--data_binary', help='location of train and test image'
#                    ' data binary to load. will be'
#                    ' created if not existing. Default'
#                    ' is {}'.format(data_bin_default),
#                    default=data_bin_default)

parser.add_argument('--data_set', help='Main directory of training and test'
                    ' data. Needs train and test subdirectories with '
                    ' images and masks subdirectories.'
                    ' Default ist {}'.format(default_data_set) +
                    ' Should have dimensions at the end. First height then'
                    ' width.',
                    default=default_data_set)

parser.add_argument('--img_height', default=192, type=int, help='Default 192')
parser.add_argument('--img_width', default=256, type=int, help='Default 256')

parser.add_argument('--model_output_choice',
                    help='Choose the last layers and activation. Choose one'
                    ' of: {}'.format(output_choices),
                    default='sigmoid')

parser.add_argument('--counter_overfit',
                    help='Chose one of {}.'.format(
                        list(counter_overfit.keys())) +
                    ' Default is reg',
                    default='reg')

parser.add_argument('--loss',
                    help='Chose one of {}'.format(list(loss_dict.keys())) +
                    ' Default is {}'.format('binary_crossentropy'),
                    default='binary_crossentropy')

parser.add_argument('--epochs',
                    help='Default is {}'.format(epochs_default),
                    type=int,
                    default=epochs_default)

parser.add_argument('--batch_size',
                    help='Default is {}'.format(batch_size_default),
                    type=int,
                    default=batch_size_default)

parser.add_argument('--do_not_use_mult_gpu',
                    help='If not chosen, multiple gpu will be used if'
                    ' available.',
                    action='store_true')

parser.add_argument('--manuell',
                    help='Choose the output image dataset manuelly (1) \
                    or go through every prepared Dataset (0). Default is {}'.format(
                        manuell_default),
                    type=int,
                    default=manuell_default)
                    
args = parser.parse_args()

#trainxy_testxy_binary =  args.data_binary
epochs = args.epochs
batch_size = args.batch_size
save_dir = args.save_dir
model_save_name = args.model_save_name
do_not_use_mult_gpu = args.do_not_use_mult_gpu
manuell = args.manuell
data_set = args.data_set

loss_function = binary_crossentropy

print("epochs ", epochs)
print("batch_size ", batch_size)
print("save_dir ", save_dir)
print("model_save_name ", model_save_name)
print("do_not_use_mult_gpu ", do_not_use_mult_gpu)
print("manuell ", manuell)
print("data_set ", data_set)

### END SCRIPT PARAMETERS ###

img_height = args.img_height
img_width  = args.img_width
img_channels = 3

print('Image shape for model is {} {} {}'.format(
    img_height, img_width, img_channels))

print("load model:" , args.load_model)

print("train on dataset:" , data_set)

target_size = (img_height, img_width)

train_generator, test_generator, train_length, test_length \
 = Unet_Data_Generator(data_set,
                         target_size, 
                         batch_size)

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
  
if args.load_model:
  template_model = models.load_model(
                      args.load_model,
                      custom_objects={'DivByRedSum': DivByRedSum,
                                      'mean_pred' : mean_pred,
                                      'max_pred': max_pred,
                                      'min_pred' : min_pred,
                                      'mean_true': mean_true,
                                      'max_true': max_true,
                                      'min_true':min_true},
                      compile=True)
  print('Loaded model from {}'.format(args.load_model))
else:
  print('Using new model with {}'.format(args.model_output_choice))
  template_model = vgg16_enc_dec_model(
                              (img_height, img_width, img_channels),
                              freeze_vgg = False,
                              output_choice =  args.model_output_choice,
                              **counter_overfit[args.counter_overfit])

n_gpu = len(get_available_gpus())
print('Number of gpu to use: {}'.format(n_gpu))
  
if n_gpu > 1 and not do_not_use_mult_gpu:
  model = multi_gpu_model(template_model, gpus=n_gpu)
  print("Training using multiple GPUs..")
else:
  model = template_model
  print("Training using single GPU or CPU..")
  
print('Using {} as loss function'.format(loss_dict[args.loss]))
model.compile(optimizer=keras.optimizers.Adam(clipnorm=1.0),
              loss=loss_dict[args.loss],
              metrics=[metrics.mae,bce_dice_loss])

#training the model
#create_dir(save_dir+"/"+model_save_name)         
create_dir(save_dir+"/"+model_save_name+"/weights")    

save_model_weights_path = save_dir+"/" +\
 model_save_name + "/weights/" + model_save_name+'_weights.hdf5'
 
print("save model weights" + save_model_weights_path) 

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 10, restore_best_weights=True)


history = model.fit_generator(
            train_generator,
            validation_data = test_generator,
            #validation_steps = test_length,
            validation_steps = 2,#for debug
            #steps_per_epoch = train_length,
            steps_per_epoch = 2, #for debug
            callbacks = [ModelCheckpoint(filepath=save_model_weights_path,
                                         verbose=1,
                                         save_best_only=True),es],
            #epochs = args.epochs
            epochs = 100
            
            ).history
               
   
                           
with open(save_dir+"/"+model_save_name+"/"+'trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history, file_pi)
    
if n_gpu > 1:
  template_model.save(save_dir+"/"+model_save_name+"/" + model_save_name + ".h5")
else:
  model.save(save_dir+"/"+model_save_name+"/" + model_save_name + ".h5")
#print('saved model at {}'.format(model_save_name))


plot_history_bce_dice_loss(history, save_dir+"/"+model_save_name)
plot_history_accuracy(history, save_dir+"/"+model_save_name)
plot_history_loss(history, save_dir+"/"+model_save_name)


#
#for n, directory_name in enumerate(dir_names):
#  print(str(n) + "   " + directory_name)
#    
#    testing_data_dir = data_set+directory_name
#    
#    test_dir = output_dir+'/'+directory_name+"_created_images"
#    
#    train_generator, test_generator, train_length, test_length =  \
#    Unet_Data_Generator(testing_data_dir,
#                         target_size, 
#                         batch_size)    
#



#
#print("MANUELL: ", manuell) 
##
#if (manuell==1):
#  while(1):
#  
#    dir_names = next(os.walk(data_set))[1]
#    
#    for n, directory_name in enumerate(dir_names):
#      print(str(n) + "   " + str(directory_name ) )
#      
#    print("e for Exit.")  
#    n = input("Enter the testing-directory number:")  
#    
#    if n == "e":
#      break
#    
#    testing_data_dir = data_set+dir_names[int(n)]
#    
#    test_dir = output_dir+'/'+dir_names[int(n)]+"_created_images"
#    
#    train_generator, test_generator, train_length, test_length =  \
#    Unet_Data_Generator(testing_data_dir,
#                         target_size, 
#                         batch_size)   
#elif (manuell ==0):
#  print("The model is predicting though every prepared dataset")
#  
#  for n, directory_name in enumerate(dir_names):
#    print(str(n) + "   " + directory_name)
#    
#    testing_data_dir = data_set+directory_name
#    
#    test_dir = output_dir+'/'+directory_name+"_created_images"
#    
#    train_generator, test_generator, train_length, test_length =  \
#    Unet_Data_Generator(testing_data_dir,
#                         target_size, 
#                         batch_size)    
#
#
#
#



