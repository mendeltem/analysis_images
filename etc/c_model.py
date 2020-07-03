#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:53:52 2019

@author: mendel,dd
"""



# Import all the necessary libraries

import matplotlib.pyplot as plt
import skimage.io                                     #Used for imshow function
import skimage.transform                              #Used for resize function
from skimage.morphology import label                  #Used for Run-Length-Encoding RLE to create final submission

import numpy as np
import pandas as pd

import keras
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv2DTranspose,Concatenate
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda,UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.merge import add, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import multi_gpu_model, plot_model
from keras import backend as K
import tensorflow as tf

from sklearn.model_selection import train_test_split


from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19

from keras import applications
from keras.models import Sequential




def get_c_model(img_width = 1024, img_height =768):

    IN_SHAPE = (* (img_width, img_height), 3)
  
    dummy= VGG19(include_top=False,input_shape=IN_SHAPE,
                  weights='imagenet')
  
    dummy.layers.pop()
    dummy.layers.pop()
    dummy.layers.pop()
    dummy.layers.pop()
    x = Conv2D(256, (2,2), activation='relu', padding='same')(dummy.layers[-1].output)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(dummy.input, decoded)
    
  
  
    return autoencoder


#img_width = 1024
#img_height =768
#IN_SHAPE = (* (img_width, img_height), 3)
#
#dummy= VGG19(include_top=False,input_shape=IN_SHAPE,
#              weights='imagenet')
#
#dummy.layers.pop()
#dummy.layers.pop()
#dummy.layers.pop()
#dummy.layers.pop()
#x = Conv2D(256, (2,2), activation='relu', padding='same')(dummy.layers[-1].output)
#x = UpSampling2D((2, 2))(x)
#x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
#x = Conv2D(64, (1, 1), activation='relu')(x)
#x = UpSampling2D((2, 2))(x)
#x = Conv2D(32, (1, 1), activation='relu')(x)
#x = UpSampling2D((2, 2))(x)
#decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
#autoencoder = Model(dummy.input, decoded)
#
#
#
#
#autoencoder.summary()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#n_ch_exps = [4, 5, 6, 7, 8, 9]   #the n-th deep channel's exponent i.e. 2**n 16,32,64,128,256
#k_size = (3, 3)                  #size of filter kernel
#k_init = 'he_normal'             #kernel initializer
#
#if K.image_data_format() == 'channels_first':
#    ch_axis = 1
#    input_shape = (3, img_width, img_height)
#elif K.image_data_format() == 'channels_last':
#    ch_axis = 3
#    input_shape = (img_width, img_height, 3)
#
#inp = Input(shape=input_shape)
#encodeds = []
#
### encoder
##enc = inp
##print(n_ch_exps)
##for l_idx, n_ch in enumerate(n_ch_exps):
##    enc = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(enc)
##    enc = Dropout(0.1*l_idx,)(enc)
##    enc = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(enc)
##    encodeds.append(enc)
##    #print(l_idx, enc)
##    if n_ch < n_ch_exps[-1]:  #do not run max pooling on the last encoding/downsampling step
##        enc = MaxPooling2D(pool_size=(2,2))(enc)
#
#
#enc = VGG19(include_top=False,
#                               input_shape=IN_SHAPE,
#                               weights='imagenet'
#                               )
#
#
#enc.summary()
#
## decoder
#dec = enc
#print(n_ch_exps[::-1][1:])
#decoder_n_chs = n_ch_exps[::-1][1:]
#for l_idx, n_ch in enumerate(decoder_n_chs):
#    l_idx_rev = len(n_ch_exps) - l_idx - 2  #
#    dec = Conv2DTranspose(filters=2**n_ch, kernel_size=k_size, strides=(2,2), activation='relu', padding='same', kernel_initializer=k_init)(dec)
#    dec = concatenate([dec, encodeds[l_idx_rev]], axis=ch_axis)
#    dec = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(dec)
#    dec = Dropout(0.1*l_idx)(dec)
#    dec = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(dec)
#
#outp = Conv2DTranspose(filters=1, kernel_size=k_size, activation='sigmoid', padding='same', kernel_initializer='glorot_normal')(dec)
#
#model = Model(inputs=[inp], outputs=[outp])
#
#
#
#
#
#









