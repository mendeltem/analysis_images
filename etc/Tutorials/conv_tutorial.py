#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:30:17 2019

@author: mendel
"""

import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

TRAIN_DIR = os.getcwd()+"/dataset/color_memory/original/images/x"
TEST_DIR = os.getcwd()+"/dataset/color_memory/original/images/v"
IMG_SIZE = 50
LR = 1e-3


MODEL_NAME = "model-{}-{}.model".format(LR,"conv")


training_data = []

for img in tqdm(os.listdir(TRAIN_DIR)):
  path = os.path.join(TRAIN_DIR,img)
  image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  #print(image.shape)
  training_data.append(image)



