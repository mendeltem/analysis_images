#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:27:47 2019

@author: mendel
"""



import sys
import pickle
import keras as ks
import tensorflow as tf
from keras.utils import to_categorical
from keras import models
from keras import layers
import os
#import exp_library as ex
import pandas as pd
import numpy as np

from modules.library import get_img_paths,get_feature_paths,memory_usage,get_csv_paths
from modules.library import getsize

from time import time
from keras.applications.inception_v3 import preprocess_input
from modules.generators import ImageSequence

import json

from os import listdir
from os.path import isfile, join

path = '/mnt/data/DATA/dataset/FEATURES/saved'

list_of_features = []

feature_paths = get_csv_paths(path)


for feature_path in feature_paths:


  feature_df = pd.read_csv(feature_path)
  
  
  name = feature_path.rsplit('/', 1)[-1]
  
  feature_df = feature_df.rename(columns = {"Unnamed: 0": name,
                                            "0": "ImageId",
                                            
                                            }) 
  
  list_of_features.append(feature_df)



#feature_df.iloc[:,2].mean()







path_t = "/mnt/data/DATA/dataset/FEATURES/saved_features_from_pictures/exception/Output_per_1.h5"

df = pd.read_csv(path_t)
