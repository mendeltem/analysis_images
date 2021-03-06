#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:53:52 2019

@author: mut,dd
"""



# Import all the necessary libraries
import os
import datetime
import glob
import random
import sys

import matplotlib.pyplot as plt
import skimage.io                                     #Used for imshow function
import skimage.transform                              #Used for resize function
from skimage.morphology import label                  #Used for Run-Length-Encoding RLE to create final submission

import numpy as np
import pandas as pd

import tensorflow as tf

keras = tf.keras


Input = keras.layers.Input
Dense = keras.layers.Dense
Activation = keras.layers.Activation
Conv2D = keras.layers.Conv2D
Conv2DTranspose = keras.layers.Conv2DTranspose
MaxPooling2D  = keras.layers.MaxPooling2D
Dropout = keras.layers.Dropout
Model = keras.models.Model
ImageDataGenerator= keras.preprocessing.image.ImageDataGenerator
concatenate = keras.layers.concatenate
ModelCheckpoint = keras.callbacks.ModelCheckpoint
VGG19 = keras.applications.VGG19
BatchNormalization = keras.layers.BatchNormalization



#from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv2DTranspose
#from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda,Reshape
#from keras.layers.advanced_activations import LeakyReLU
#from keras.models import load_model, Model
#from keras.preprocessing.image import ImageDataGenerator
#from keras.layers.merge import add, concatenate
#from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.utils import multi_gpu_model, plot_model
#from keras import backend as K
#from keras.applications import VGG19
#from keras import regularizers
#
#
#
#from keras.layers import Concatenate
#Input = keras.layer.Input

from sklearn.model_selection import train_test_split
from modules.model_zoo import bce_loss, dice_loss, bce_dice_loss
from modules.model_zoo import precision, recall
kl_divergence = keras.losses.kullback_leibler_divergence #TODO: use this


loss = 'bce_dice_loss'
metrics = ['dice_loss']

#dict to make variable selection simpler
metrics_and_losses = {'dice_loss':dice_loss,
                       'bce_dice_loss':bce_dice_loss,
                       'bce_loss':bce_loss,
                       'precision': precision,
                       'recall': recall}


## Design our model architecture here
#def keras_model(img_width=1024, img_height=768):
#    '''returns compiled unet model
#    Modified from https://keunwoochoi.wordpress.com/2017/10/11/u-net-on-keras-2-0/
#    '''
#    n_ch_exps = [4, 5, 6, 7, 8, 9]   #the n-th deep channel's exponent i.e. 2**n 16,32,64,128,256
#    k_size = (3, 3)                  #size of filter kernel
#    k_init = 'he_normal'             #kernel initializer
#
#    if K.image_data_format() == 'channels_first':
#        ch_axis = 1
#        input_shape = (3, img_width, img_height)
#    elif K.image_data_format() == 'channels_last':
#        ch_axis = 3
#        input_shape = (img_width, img_height, 3)
#
#    inp = Input(shape=input_shape)
#    encodeds = []
#
#    # encoder
#    enc = inp
#    print(n_ch_exps)
#    for l_idx, n_ch in enumerate(n_ch_exps):
#        enc = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(enc)
#        enc = Dropout(0.1*l_idx,)(enc)
#        enc = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(enc)
#        encodeds.append(enc)
#        #print(l_idx, enc)
#        if n_ch < n_ch_exps[-1]:  #do not run max pooling on the last encoding/downsampling step
#            enc = MaxPooling2D(pool_size=(2,2))(enc)
#
#    # decoder
#    dec = enc
#    print(n_ch_exps[::-1][1:])
#    decoder_n_chs = n_ch_exps[::-1][1:]
#    for l_idx, n_ch in enumerate(decoder_n_chs):
#        l_idx_rev = len(n_ch_exps) - l_idx - 2  #
#        dec = Conv2DTranspose(filters=2**n_ch, kernel_size=k_size, strides=(2,2), activation='relu', padding='same', kernel_initializer=k_init)(dec)
#        dec = concatenate([dec, encodeds[l_idx_rev]], axis=ch_axis)
#        dec = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(dec)
#        dec = Dropout(0.1*l_idx)(dec)
#        dec = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(dec)
#
#    outp = Conv2DTranspose(filters=1, kernel_size=k_size, activation='sigmoid', padding='same', kernel_initializer='glorot_normal')(dec)
#
#    model = Model(inputs=[inp], outputs=[outp])
#
#    model.compile(optimizer='adam',
#                  loss=metrics_and_losses[loss],
#                  metrics=[metrics_and_losses[m] for m in metrics])
#
#    return model
#

#
#
#def keras_dmodel(img_width=1024, img_height=768):
#  '''returns compiled unet model with pretrained VGG19 Conv Net
#  Modified from https://keunwoochoi.wordpress.com/2017/10/11/u-net-on-keras-2-0/
#  '''
#
#  #img_width=1024
#  #img_height=768
#  input_shape = (img_width,img_height, 3)
#
#  k_size = (2, 2)                  #size of filter kernel
#  k_init = 'he_normal'
#
#  #encoder
#  enc= VGG19(include_top=False,input_shape=input_shape,
#                weights='imagenet')
#
#  for i,layer in enumerate(enc.layers):
#    print(i," ",layer.output)
#
#
#  down4_pool = enc.layers[16].output
#
#  up = Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer=k_init)(down4_pool)
#  up = BatchNormalization()(up)
#  up = Dropout(0.1,)(up)
#  up = MaxPooling2D(pool_size=(2,2))(up)
#
#  up = Conv2D(filters=1024, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer=k_init)(up)
#  up = BatchNormalization()(up)
#
#  up = Dropout(0.1,)(up)
#  up = Dense(1024)(up)
#  up = Dropout(0.1,)(up)
#  up = Dense(1024)(up)
#
#  up = Activation('relu')(up)
#  up = BatchNormalization()(up)
#  up = Dropout(0.1,)(up)
#
#  up = Conv2DTranspose(filters=512, kernel_size=k_size, strides=(2,2),
#                      activation='relu',
#                      padding='same', kernel_initializer=k_init)(up)
#  up = BatchNormalization()(up)
#
#  up = concatenate([up, enc.layers[16].output], axis=3)
#
#  up = Conv2D(512, kernel_size=(3,3), activation='relu', padding='same',
#             kernel_initializer=k_init)(up)
#
#  up = BatchNormalization()(up)
#  up = Dropout(0.1,)(up)
#
#  up = Conv2DTranspose(filters=256, kernel_size=k_size, strides=(2,2),
#                      activation='relu',
#                      padding='same', kernel_initializer=k_init)(up)
#  up = BatchNormalization()(up)
#
#  up = concatenate([up, enc.layers[11].output], axis=3)
#
#  up = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same',
#             kernel_initializer=k_init)(up)
#
#  up = BatchNormalization()(up)
#  up = Dropout(0.1,)(up)
#
#  up = Conv2DTranspose(filters=128, kernel_size=k_size, strides=(2,2),
#                      activation='relu', padding='same',
#                      kernel_initializer=k_init)(up)
#  up = BatchNormalization()(up)
#
#  up = concatenate([up, enc.layers[6].output], axis=3)
#
#  up = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same',
#             kernel_initializer=k_init)(up)
#  up = BatchNormalization()(up)
#
#  up = Dropout(0.1*2,)(up)
#
#  up = Conv2DTranspose(filters=64, kernel_size=k_size, strides=(2,2),
#                      activation='relu', padding='same',
#                      kernel_initializer=k_init)(up)
#  up = BatchNormalization()(up)
#
#  up = concatenate([up, enc.layers[5].output], axis=3)
#
#  up = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same',
#             kernel_initializer=k_init)(up)
#  up = BatchNormalization()(up)
#  up = Dropout(0.1*3,)(up)
#
#  up = Conv2DTranspose(filters=32, kernel_size=k_size, strides=(2,2),
#                      activation='relu', padding='same',
#                      kernel_initializer=k_init)(up)
#  up = BatchNormalization()(up)
#  up = concatenate([up, enc.layers[1].output], axis=3)
#
#  dec = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(up)
#  up = BatchNormalization()(up)
#
#  autoencoder = Model(inputs=[enc.input], outputs=[dec])
#
#  #autoencoder.summary()
#  for i,layer in enumerate(autoencoder.layers):
#    if i < 17:
#      layer.trainable = False
#
#  autoencoder.compile(optimizer='adam',
#              loss=metrics_and_losses[loss],
#              metrics=[metrics_and_losses[m] for m in metrics])
#  return autoencoder
##
#
#def keras_dmodel(img_width=1024, img_height=768):
#  '''returns compiled unet model with pretrained VGG19 Conv Net
#  Modified from https://keunwoochoi.wordpress.com/2017/10/11/u-net-on-keras-2-0/
#  '''
#  input_shape = (img_width,img_height, 3)
#
#  #n_ch_exps = [ 5, 6, 7, 8, 9]   #the n-th deep channel's exponent i.e. 2**n 16,32,64,128,256
#  k_size = (2, 2)                  #size of filter kernel
#  k_init = 'he_normal'
#
#
#  #encoder
#  enc= VGG19(include_top=False,input_shape=input_shape,
#                weights='imagenet')
#
#  x = enc.layers[-6].output
#
#  for i,layer in enumerate(enc.layers):
#    print(i,layer.output)
#
#  #conv2_1_shape =  enc.layers[4].input_shape
#
#  #l1 = tf.keras.backend.resize_images(enc.layers[17], conv2_1_shape[2],conv2_1_shape[1], data_format = "channels_first")
#
#
#  #x = Dense(256, activation='relu',activity_regularizer=regularizers.l1(10e-5))(x)
#
#  x = Conv2DTranspose(filters=256, kernel_size=k_size, strides=(2,2),
#                      activation='relu',
#                      padding='same', kernel_initializer=k_init)(x)
#  #x = concatenate([x, enc.layers[15].output], axis=3)
#
#
#  up0 = concatenate([down0, up0], axis=3)
#
#
#
#  x = Conv2D(256, kernel_size=k_size, activation='relu', padding='same',
#             kernel_initializer=k_init)(x)
#  x = BatchNormalization()(x)
#  x = Dropout(0.1,)(x)
#
#  x = Conv2DTranspose(filters=128, kernel_size=k_size, strides=(2,2),
#                      activation='relu', padding='same',
#                      kernel_initializer=k_init)(x)
#  x = Conv2D(128, kernel_size=k_size, activation='relu', padding='same',
#             kernel_initializer=k_init)(x)
#  x = BatchNormalization()(x)
#  x = Dropout(0.1*2,)(x)
#
#  x = Conv2DTranspose(filters=64, kernel_size=k_size, strides=(2,2),
#                      activation='relu', padding='same',
#                      kernel_initializer=k_init)(x)
#  x = Conv2D(64, kernel_size=k_size, activation='relu', padding='same',
#             kernel_initializer=k_init)(x)
#  x = BatchNormalization()(x)
#  x = Dropout(0.1*3,)(x)
#
#  x = Conv2DTranspose(filters=32, kernel_size=k_size, strides=(2,2),
#                      activation='relu', padding='same',
#                      kernel_initializer=k_init)(x)
#  x = BatchNormalization()(x)
#  dec = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
#
#
#  autoencoder = Model(inputs=[enc.input], outputs=[dec])
#
#  autoencoder.summary()
#  for i,layer in enumerate(autoencoder.layers):
#    if i < 17:
#      layer.trainable = True
#
#  autoencoder.compile(optimizer='adam',
#              loss=metrics_and_losses[loss],
#              metrics=[metrics_and_losses[m] for m in metrics])
#
#  return autoencoder
##autoencoder.summary()
##
##
##for i,layer in enumerate(enc.layers):
##  print(i," ",layer.output)
#def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
#    """Function to add 2 convolutional layers with the parameters passed to it"""
#    # first layer
#    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
#              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
#    if batchnorm:
#        x = BatchNormalization()(x)
#    x = Activation('relu')(x)
#
#    # second layer
#    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
#              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
#    if batchnorm:
#        x = BatchNormalization()(x)
#    x = Activation('relu')(x)
#
#    return x

def keras_cmodel(img_width=1024, img_height=768):
  '''returns compiled unet model with pretrained VGG19 Conv Net
  Modified from https://keunwoochoi.wordpress.com/2017/10/11/u-net-on-keras-2-0/
  '''

  #img_width=1024
  #img_height=768
  input_shape = (img_width,img_height, 3)

  k_size = (2, 2)                  #size of filter kernel
  k_init = 'he_normal'

  #encoder
  enc= VGG19(include_top=False,input_shape=input_shape,
                weights='imagenet')

#  for i,layer in enumerate(enc.layers):
#    print(i," ",layer.output)
  down4_pool = enc.layers[16].output
#
#  for i, en in enumerate(enc.layers):
#    print(i," ",en.output)
##
#
  up = Conv2DTranspose(filters=256, kernel_size=k_size, strides=(2,2),
                      activation='relu',
                      padding='same', kernel_initializer=k_init)(down4_pool)

  up = concatenate([up, enc.layers[11].output], axis=3)
  up = Dropout(0.1)(up)


  up = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same',
             kernel_initializer=k_init)(up)

  up = BatchNormalization()(up)


  up = Conv2DTranspose(filters=128, kernel_size=k_size, strides=(2,2),
                      activation='relu', padding='same',
                      kernel_initializer=k_init)(up)

  up = concatenate([up, enc.layers[6].output], axis=3)
  up = Dropout(0.1)(up)
  up = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same',
             kernel_initializer=k_init)(up)
  #up = Conv2D(128, (3, 3), activation='relu', padding='same')(up)
  #up = Conv2D(128, (3, 3), activation='relu', padding='same')(up)

  up = Conv2DTranspose(filters=64, kernel_size=k_size, strides=(2,2),
                      activation='relu', padding='same',
                      kernel_initializer=k_init)(up)

  up = concatenate([up, enc.layers[5].output])
  up = Dropout(0.1)(up)

  up = Conv2DTranspose(filters=32, kernel_size=k_size, strides=(2,2),
                      activation='relu', padding='same',
                      kernel_initializer=k_init)(up)
  up = Conv2D(32, (3, 3), activation='relu', padding='same')(up)
  up = Conv2D(32, (3, 3), activation='relu', padding='same')(up)
  dec = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(up)
#TODO:  dec = dec/"SUMME der werte"

  autoencoder = Model(inputs=[enc.input], outputs=[dec])

  #autoencoder.summary()
  for i,layer in enumerate(autoencoder.layers):
    if i < 17:
      layer.trainable = False

  autoencoder.compile(optimizer='adam',
              loss=kl_divergence,
              metrics=[metrics_and_losses[m] for m in metrics])
  return autoencoder

sum_tensor = keras.backend.sum
K = keras.backend
#
#
def keras_cmodel_beta(img_width=1024, img_height=768):
  '''returns compiled unet model with pretrained VGG19 Conv Net
  Modified from https://keunwoochoi.wordpress.com/2017/10/11/u-net-on-keras-2-0/
  '''

  #img_width=1024
  #img_height=768
  input_shape = (img_width,img_height, 3)

  k_size = (2, 2)                  #size of filter kernel
  k_init = 'he_normal'

  #encoder
  enc= VGG19(include_top=False,input_shape=input_shape,
                weights='imagenet')

#  for i,layer in enumerate(enc.layers):
#    print(i," ",layer.output)
  down4_pool = enc.layers[16].output
#
#  for i, en in enumerate(enc.layers):
#    print(i," ",en.output)
##
#
  up = Conv2DTranspose(filters=256, kernel_size=k_size, strides=(2,2),
                      activation='relu',
                      padding='same', kernel_initializer=k_init)(down4_pool)
  up = BatchNormalization()(up)
  up = concatenate([up, enc.layers[11].output], axis=3)
  up = BatchNormalization()(up)
  up = Dropout(0.1)(up)
  up = Conv2D(256, (3,3), activation='relu', padding='same')(up)
  up = BatchNormalization()(up)
  up = Conv2D(256, (3,3), activation='relu', padding='same')(up)
  up = BatchNormalization()(up)

  up = Conv2DTranspose(filters=128, kernel_size=k_size, strides=(2,2),
                      activation='relu', padding='same',
                      kernel_initializer=k_init)(up)
  up = BatchNormalization()(up)
  up = concatenate([up, enc.layers[6].output], axis=3)
  up = BatchNormalization()(up)
  up = Dropout(0.1)(up)
  up = Conv2D(128, (3,3), activation='relu', padding='same')(up)
  up = BatchNormalization()(up)
  up = Conv2D(128, (3,3), activation='relu', padding='same')(up)
  up = BatchNormalization()(up)

  up = Conv2DTranspose(filters=64, kernel_size=k_size, strides=(2,2),
                      activation='relu', padding='same',
                      kernel_initializer=k_init)(up)
  up = BatchNormalization()(up)
  up = Conv2D(64, (3, 3), activation='relu', padding='same')(up)
  up = BatchNormalization()(up)
  up = Conv2D(64, (3, 3), activation='relu', padding='same')(up)
  up = BatchNormalization()(up)
  up = concatenate([up, enc.layers[5].output])
  up = BatchNormalization()(up)
  up = Dropout(0.1)(up)

  up = Conv2DTranspose(filters=32, kernel_size=k_size, strides=(2,2),
                      activation='relu', padding='same',
                      kernel_initializer=k_init)(up)
  up = BatchNormalization()(up)
  up = Conv2D(32, (3, 3), activation='relu', padding='same')(up)
  up = BatchNormalization()(up)
  up = Conv2D(32, (3, 3), activation='relu', padding='same')(up)
  up = BatchNormalization()(up)
  dec = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(up)


  dec = BatchNormalization()(dec)


  model = Model(inputs=[enc.input], outputs=[dec])

  #autoencoder.summary()
  for i,layer in enumerate(model.layers):
    if i < 17:
      layer.trainable = False

  model.compile(optimizer='adam',
              loss=kl_divergence,
              metrics=['accuracy'])
  return model
#
#
##def conv_layer(
##        self, bottom, in_channels,
##        out_channels, name, batchnorm=None, filter_size=3):
##    with tf.variable_scope(name):
##        filt, conv_biases = self.get_conv_var(
##            filter_size, in_channels, out_channels, name)
##
##        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
##        bias = tf.nn.bias_add(conv, conv_biases)
##        relu = tf.nn.relu(bias)
##
##        # if batchnorm is not None:
##        #     if name in batchnorm:
##        #         relu = self.batchnorm(relu)
##
##        return relu, bias
##



def keras_bmodel(img_width=1024, img_height=768):
  '''returns compiled unet model with pretrained VGG19 Conv Net
  Modified from https://keunwoochoi.wordpress.com/2017/10/11/u-net-on-keras-2-0/
  '''
  #img_width=1024
  #img_height=768
  #input_shape = (img_width,img_height, 3)

  k_size = (1, 1)                  #size of filter kernel
  k_init = 'he_normal'

  #encoder
  vgg19= VGG19(include_top=False,input_shape=(img_width,img_height, 3),
                weights='imagenet')

  for i,layer in enumerate(vgg19.layers):
    print(i," ",layer.output)

  conv2_1= vgg19.layers[4].output

   #get the shape of conv2_1
#  resize_size = [int(x) for i, x in enumerate(conv2_1.output.get_shape()) if i != 0]
#  new_size = np.asarray([resize_size[0], resize_size[1]])

#  resized = tf.image.resize_images(vgg19.layers[17].output,new_size)

  l1 = Conv2DTranspose(filters=128, kernel_size=(5,5), strides=(8,8),
                      activation='relu', padding='same',
                      kernel_initializer=k_init)(vgg19.layers[17].output)

  l2 = Conv2DTranspose(filters=128, kernel_size=(5,5), strides=(8,8),
                      activation='relu', padding='same',
                      kernel_initializer=k_init)(vgg19.layers[18].output)

  l3 = Conv2DTranspose(filters=128, kernel_size=(5,5), strides=(8,8),
                      activation='relu', padding='same',
                      kernel_initializer=k_init)(vgg19.layers[19].output)

  l4 = Conv2DTranspose(filters=128, kernel_size=(5,5), strides=(8,8),
                    activation='relu', padding='same',
                    kernel_initializer=k_init)(vgg19.layers[20].output)

  l5 = Conv2DTranspose(filters=128, kernel_size=(5,5), strides=(16,16),
                    activation='relu', padding='same',
                    kernel_initializer=k_init)(vgg19.layers[21].output)

  #up = concatenate([l1, l2,l3,l4,l5], axis=3)


  up = concatenate([l3,l4], axis=3)

  upb = BatchNormalization()(up)

  upb = Dropout(0.1)(upb)


  #TODO Drop out during training

  convt = Conv2DTranspose(filters=128, kernel_size=(5,5), strides=(2,2),
                    activation='relu', padding='same',
                    kernel_initializer=k_init)(upb)


  conv1 = Conv2D(16, kernel_size=(1,1), activation='relu', padding='same',
         kernel_initializer=k_init)(convt)

  conv1d = Dropout(0.1)(conv1)
  conv1b = BatchNormalization()(conv1d)


  conv2 = Conv2D(32, kernel_size=(1,1), activation='relu', padding='same',
         kernel_initializer=k_init)(conv1b)

  conv2d = Dropout(0.1)(conv2)
  conv2b = BatchNormalization()(conv2d)

  conv3 = Conv2D(4, kernel_size=(1,1), activation='relu', padding='same',
         kernel_initializer=k_init)(conv2b)

  conv3d = Dropout(0.1)(conv3)
  conv3b = BatchNormalization()(conv3d)
#  conv2 = Conv2D(128, kernel_size=k_size, activation='relu', padding='same',
#         kernel_initializer=k_init)(conv1)

  #conv2 = Dropout(0.1)(conv2)
  #conv2 = BatchNormalization()(conv2)


  out = Conv2D(1, kernel_size=(1,1), activation='sigmoid', padding='same',
         kernel_initializer=k_init)(conv3b)

  deepgaze = Model(inputs=[vgg19.input], outputs=[out])

  deepgaze.summary()

  #autoencoder.summary()
  for i,layer in enumerate(deepgaze.layers):
    if i < 17:
      layer.trainable = False

  deepgaze.compile(optimizer='adam',
              loss=metrics_and_losses[loss],
              metrics=[metrics_and_losses[m] for m in metrics])
  return deepgaze


