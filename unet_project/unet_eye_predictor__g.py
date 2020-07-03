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
import pickle
import platform

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import models
from tensorflow.python.ops import math_ops

from modules.m_preprocess import load_images_and_masks, image_mask_generator
from modules.unet_vgg16 import vgg16_enc_dec_model, DivByRedSum, output_choices
from modules.mult_gpu import get_available_gpus

metrics = keras.metrics
ModelCheckpoint = keras.callbacks.ModelCheckpoint
kl_divergence = keras.losses.kullback_leibler_divergence
mean_squared_error = keras.losses.mean_squared_error
binary_crossentropy = keras.losses.binary_crossentropy
multi_gpu_model = keras.utils.multi_gpu_model


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
counter_overfit = {'reg' : {'reg':0.001,'dropout':None},
                   'dropout' : {'reg':None,'dropout':0.1},
                   'none' : {'reg':None,'dropout':None}}

main_data_dir = 'dataset/Set_01'
data_bin_default = 'save/trainxy_testxy_raw_192_256.p'
save_dir = 'save/vgg16_unet'
model_save_name = 'trained_model.h5'

batch_size_default = 8
epochs_default = 50

seed = 67
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

parser.add_argument('--data_binary', help='location of train and test image'
                    ' data binary to load. will be'
                    ' created if not existing. Default'
                    ' is {}'.format(data_bin_default),
                    default=data_bin_default)

parser.add_argument('--main_data_dir', help='Main directory of training and test'
                    ' data. Needs train and test subdirectories with '
                    ' images and masks subdirectories.'
                    ' Default ist {}'.format(main_data_dir) +
                    ' Should have dimensions at the end. First height then'
                    ' width.',
                    default=main_data_dir)

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

args = parser.parse_args()

trainxy_testxy_binary =  args.data_binary
main_data_dir = args.main_data_dir
epochs = args.epochs
batch_size = args.batch_size
save_dir = args.save_dir
model_save_name = args.model_save_name
do_not_use_mult_gpu = args.do_not_use_mult_gpu

<<<<<<< HEAD
<<<<<<< HEAD:unet_eye_predictor.py
loss_function = binary_crossentropy
### END SCRIPT PARAMETERS ###


img_height = int(img_base_height/down_size)
img_width  = int(img_base_width/down_size)
img_channels = img_base_channels
=======
=======
<<<<<<< HEAD:m_unet_eye_predictor.py
>>>>>>> david_copy
img_height = args.img_height
img_width  = args.img_width
img_channels = 3

print('Image shape for model is {} {} {}'.format(
    img_height, img_width, img_channels))

print('Using batch size: {}'.format(batch_size))
<<<<<<< HEAD
>>>>>>> david_copy:m_unet_eye_predictor.py
=======
=======
loss_function = binary_crossentropy
### END SCRIPT PARAMETERS ###


img_height = int(img_base_height/down_size)
img_width  = int(img_base_width/down_size)
img_channels = img_base_channels
>>>>>>> origin/mendel:unet_eye_predictor.py
>>>>>>> david_copy

os.makedirs(save_dir, exist_ok = True)

def check_datafname(fname, h=img_height, w=img_width):
  """Works only for three digits sizes."""
  try:
    w_correct = (int(fname[-5:-2]) == w)
    h_correct = (int(fname[-9:-6]) == h)
  except:
    return
  if (not w_correct) or (not h_correct):
    print('Width of file is {}'.format(int(fname[-5:-2])))
    print('Downscaled width for training is {}'.format(w))
    print('Height of file is {}'.format(int(fname[-9:-6])))
    print('Downscaled height for training is {}'.format(h))
    raise Warning('Data binary file name does not fit image shape. See'
                  ' image height and image width parser argument.')

#load the images and masks
ttb_path = trainxy_testxy_binary
check_datafname(ttb_path)

if os.path.exists(ttb_path):
  print('Loading binary with training and test data: ', ttb_path)
  trainX, trainY, testX, testY = pickle.load(open(ttb_path, 'rb'))
else:
  print('Could not load binary with training and test data: ',
        ttb_path)
  print('Creating and saving binary with training and test data at: {}'.format(
      ttb_path))
  trainX, trainY = load_images_and_masks(join(main_data_dir, 'train'),
                                         width = img_width,
                                         height = img_height,
                                         br = None)
  testX, testY = load_images_and_masks(join(main_data_dir, 'test'),
                                       width = img_width,
                                       height = img_height,
                                       br = None)
  pickle.dump([trainX, trainY, testX, testY], open(ttb_path, 'wb'), protocol=4)

print('Modify ground truth is {}'.format(args.model_output_choice),
      'According to model output choice.')
train_generator, test_generator = image_mask_generator(
                                   trainX,
                                   trainY,
                                   testX,
                                   testY,
                                   batch_size,
                                   modify_truth = args.model_output_choice)

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
              metrics=[metrics.mae, min_pred, max_pred, mean_pred,
                       min_true, max_true, mean_true])

#training the model
save_model_weights_path = splitext(join(save_dir, model_save_name))[0] +\
 '_weights.hdf5'
history = model.fit_generator(
            train_generator,
            validation_data = test_generator,
            validation_steps = int(len(testX)/batch_size),
            steps_per_epoch = int(len(trainX)/batch_size),
            callbacks = [ModelCheckpoint(filepath=save_model_weights_path,
                                         verbose=1,
                                         save_best_only=True)],
            epochs = args.epochs).history
if n_gpu > 1:
  template_model.save(join(save_dir, model_save_name))
else:
  model.save(join(save_dir, model_save_name))
print('saved model at {}'.format(join(save_dir, model_save_name)))

history_save_name = splitext(model_save_name)[0] + '_history.p'
pickle.dump(history, open(join(save_dir, history_save_name),'wb'), protocol=4)

plt.plot(history["val_loss"],
         linestyle='-',
         color='r',
         label='Val Loss: ' + str(round(abs(history["val_loss"][-1]), 3)))
plt.plot(history["loss"],
         linestyle='-',
         color='b',
         label='Loss: ' + str(round(history["loss"][-1], 3)))
plt.title('Training and Test Loss')
plt.legend()
<<<<<<< HEAD
<<<<<<< HEAD:unet_eye_predictor.py
plt.savefig(join(save_dir,"history.jpg"))
plt.show()
=======
plt.savefig(join(save_dir, splitext(history_save_name)[0] + '.jpg'))
>>>>>>> david_copy:m_unet_eye_predictor.py
=======
<<<<<<< HEAD:m_unet_eye_predictor.py
plt.savefig(join(save_dir, splitext(history_save_name)[0] + '.jpg'))
=======
plt.savefig(join(save_dir,"history.jpg"))
plt.show()
>>>>>>> origin/mendel:unet_eye_predictor.py
>>>>>>> david_copy
