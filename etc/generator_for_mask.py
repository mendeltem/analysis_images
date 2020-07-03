#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:06:17 2019

eye movement predictor, load the model and images from the eye movement 
experiment and predict the eye movement with kernel density estimation mask. 

@author: mut,dd
"""
import os
#set the working directory
ML_PATH ="/home/mendel/Documents/work/mendel_exchange"
os.chdir(ML_PATH)
#from scipy import misc
import numpy as np
from tensorflow import keras
ModelCheckpoint = keras.callbacks.ModelCheckpoint
import imageio
from pathlib import Path
import types
from keras.utils import Sequence
from skimage.transform import resize
import cv2
"""
generators, iterators or sequences as defined by the keras api to feed
the keras model functions.
"""

class ImageSequence(Sequence):
  """Almost exact copy of a very simple Sequence reading images
  in batches example from the keras page.
  
  Arguments:
      paths: paths to the images and masks
  
      batch_size: number of items being sent per __getitem__ call
  
      augmentation: list of functions to augment image data. list index 0
      is applied first. preprocessing happens before augmentation.
  """
  def __init__(self,
               paths,
               batch_size,
               augmentation = None,
               target_size = (256,256)):
    
    paths = Path(paths)
       
    self.images_paths =  list(paths.glob("images/*.pneg"))
    self.mask_paths   =  list(paths.glob("masks/*.pneg"))
    
    self.images_paths.sort()
    self.mask_paths.sort()

    
    self.batch_size = batch_size
    
    self.target_size = target_size
    
    if augmentation is None:
        augmentation = []
    elif isinstance(augmentation, types.FunctionType):
        augmentation = [augmentation]
    self.fn_list = augmentation
    
  def __len__(self):
      return int(np.ceil(len(self.images_paths) / float(self.batch_size)))
    
   
  def print_var(self):
    print(self.images_paths)
    
  def __getitem__(self, idx):
    batch_images = self.images_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
    batch_masks  = self.mask_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
      
    
    batch_images_read_in = []
    batch_masks_read_in  = []
    
  
    for i in range(len( batch_images)):
      try:
      
        image= imageio.imread(batch_images[i])# / 255
        mask= imageio.imread(batch_masks[i]) #/ 255
#        print("Image ", batch_images[i])
#        print("Mask ", batch_masks[i])
        
        image = cv2.normalize(image,
                       None,
                       alpha=0,
                       beta=1,
                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = resize(image,
                       (self.target_size[0], self.target_size[1], 3),
                        mode='constant',
                        preserve_range=True,
                        anti_aliasing=True,
                        anti_aliasing_sigma=None)
        mask = cv2.normalize(mask,
                             None,
                             alpha=0,
                             beta=1,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        mask = resize(mask, 
                       (self.target_size[0], self.target_size[1], 1),
                        mode='constant',
                        preserve_range=True,
                        anti_aliasing=True,
                        anti_aliasing_sigma=None)
        
        
          
        for fn in self.fn_list:
          image = fn(image)
        batch_images_read_in.append(image)
        batch_masks_read_in.append(mask)
        
      except:
        print("Error Loading")
#        print("Image ", batch_images[i])
#        print("Mask ", batch_masks[i])
      
    yield (np.array(batch_images_read_in), np.array(batch_masks_read_in))
  
  
  
  
      
















