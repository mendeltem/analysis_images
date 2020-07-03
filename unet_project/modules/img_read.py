#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:16:36 2018

@author: DD

Read image files.
"""
from skimage import io
from PIL import PngImagePlugin
#This is a setting to ensure that images with large metadata are read without
#an exception
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 2**20
import numpy as np

__all__ = ['imread_rgb']


def imread_rgb(img_path):
    """
    Reads an image file and returns a numpy array with 3 color channels.
    Image has to have uint8 values. (Not 100% sure if that is right.)
    Gray scale images get three identical rgb channels.
    Alpha channels are getting removed.
    """
    #imread can have problems with big metadata. the plugin from PIL will then
    #raise a value error --> global MAX_TEXT_CHUNK variable
    try:
        x = io.imread(img_path)
    except ValueError as e:
        raise print('Something went wrong with the Image at {}'
                  .format(img_path),e)
    if x.dtype != 'uint8':
        raise Exception("imread does not result in data type uint8.")
    # Convert gray scale to rgb
    if len(x.shape) == 2:
        y = np.zeros(x.shape + (3,), dtype=np.uint8)
        for i in range(3):
            y[:,:,i] = x
        x = y
    # Discard alpha channel
    elif x.shape[2]>3:
        x = x[:,:,0:3]
    return x