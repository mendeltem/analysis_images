#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:16:51 2018

@author: DD

generators, iterators or sequences as defined by the keras api to feed
the keras model functions.
"""
from keras.utils import Sequence
import numpy as np
import types

from modules.img_read import imread_rgb

# maybe useful imports for other Sequence child classes:
# from skimage.io import imread
from skimage.transform import resize

class ImageSequence(Sequence):
    """Almost exact copy of a very simple Sequence reading images
    in batches example from the keras page.

    Arguments:
        paths: paths to the images

        labels: labels for the images in the same order. If None, the
        sequence returns only the numpy array of the images without
        labels.

        batch_size: number of items being sent per __getitem__ call

        preprocessing: list functions to preprocess image data. list index 0
        is applied first. preprocessing happens before augmentation.

        augmentation: list of functions to augment image data. list index 0
        is applied first. preprocessing happens before augmentation.
    """

    def __init__(self,
                 paths,
                 labels,
                 batch_size,
                 preprocessing,
                 augmentation,
                 target_size = None):
        self.x, self.y = paths, labels
        self.batch_size = batch_size

        if preprocessing is None:
            preprocessing = []
        elif isinstance(preprocessing, types.FunctionType):
            preprocessing = [preprocessing]
        if augmentation is None:
            augmentation = []
        elif isinstance(augmentation, types.FunctionType):
            augmentation = [augmentation]
        self.fn_list = preprocessing + augmentation

        if target_size is not None:
            self.fn_list.append(lambda x : resize(x, target_size))

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x_read_in = []
        for file_name in batch_x:
            img = imread_rgb(file_name)
            for fn in self.fn_list:
                img = fn(img)
            batch_x_read_in.append(img)
        if self.y is not None:
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            return np.array(batch_x_read_in), np.array(batch_y)
        else:
            return np.array(batch_x_read_in)


