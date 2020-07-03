#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:41:15 2019

@author: mendel
"""

import numpy as np
import keras

class Image_DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,
                 list_IDs,
                 labels,
                 image_dir = "",
                 mask_dir = "",
                 batch_size=1,
                 Image_dim=(768,1024,32),
                 Mask_dim=(768,1024,1),
                 n_classes=10,
                 shuffle=True):
        'Initialization'
        self.Image_dim = Image_dim
        self.Mask_dim = Mask_dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.image_dir = image_dir
        self.mask_dir = mask_dir  
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.Image_dim))
        y = np.empty((self.batch_size, *self.Mask_dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store image
            X[i] = np.load(self.image_dir + ID + '.png')

            # Store mask
            y[i] = np.load(self.mask_dir + ID + '.png')

        return X, y