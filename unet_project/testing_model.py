#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:50:52 2019

@author: mendel
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
#from tensorflow import keras

import keras

from keras.models import load_model

data_set = "/mnt/data/DATA/dataset/UNET/prepared_dataset/"


output = "/mnt/data/DATA/dataset/UNET/Output/"



models_dirertorys = next(os.walk(output))[1]



models_dirertory = output + models_dirertorys[0]


def get_model_paths(start_dir, extensions = ['h5']):
    """Returns all image paths with the given extensions in the directory.

    Arguments:
        start_dir: directory the search starts from.

        extensions: extensions of image file to be recognized.

    Returns:
        a sorted list of all image paths starting from the root of the file
        system.
    """
    if start_dir is None:
        start_dir = os.getcwd()
    img_paths = []
    for roots,dirs,files in os.walk(start_dir):
        for name in files:
            for e in extensions:
                if name.endswith('.' + e):
                    img_paths.append(roots + '/' + name)
    img_paths.sort()
    return img_paths
  
  
model_paths = get_model_paths(models_dirertory)


print(model_paths[0])


model = load_model(model_paths[0])

load_model('/mnt/data/DATA/dataset/UNET/Output/ANKES_BW_25_control/ANKES_BW_25_control_4_binary_crossentropy/ANKES_BW_25_control_4_binary_crossentropy.h5')
