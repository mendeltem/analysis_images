#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:12:03 2020

@author: mendel
"""

import numpy as np
import pandas as pd
import os
import feather
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from pympler import asizeof

from keras.utils import Sequence
import keras

def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
           n_classes=10, shuffle=True):
  'Initialization'
  self.dim = dim
  self.batch_size = batch_size
  self.labels = labels
  self.list_IDs = list_IDs
  self.n_channels = n_channels
  self.n_classes = n_classes
  self.shuffle = shuffle
  self.on_epoch_end()
  
  
  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.list_IDs))
    if self.shuffle == True:
        np.random.shuffle(self.indexes)
          
          
  def __data_generation(self, list_IDs_temp):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    X = np.empty((self.batch_size, *self.dim, self.n_channels))
    y = np.empty((self.batch_size), dtype=int)
  
    # Generate data
    for i, ID in enumerate(list_IDs_temp):
        # Store sample
        X[i,] = np.load('data/' + ID + '.npy')
  
        # Store class
        y[i] = self.labels[ID]
  
    return X, keras.utils.to_categorical(y, num_classes=self.n_classes)          
  
  
  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.list_IDs) / self.batch_size))
  
  
  
  
  