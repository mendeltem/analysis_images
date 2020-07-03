#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 15:17:26 2018

@author: DD, UT

compute and save intermediate representations of fixations on images from the
inception_v3 net.
"""

import sys
import pickle
import keras
import os
import gc
#import exp_library as ex
import pandas as pd
import numpy as np
import re
from time import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


from keras.applications.inception_v3 import preprocess_input
from modules.generators import ImageSequence

DEFAULT_SAVE_PATH_MODEL     = 'model_inception_v3_mixedlayeroutputs_auto.h5'
DEFAULT_IMAGE_DIRECTORY     = 'mem_images/'
DEFAULT_SAVE_DATA_PATH      = 'activations_of_fixations_auto.p' #TODO: think about a good name maybe
DEFAULT_EYE_FIXATION_DAT    = "finalresulttab_funcsac_SFC_memory.dat"
DEFAULT_EXPERIMENT          = "memory"


all_data_uncleared = pd.read_table(DEFAULT_EYE_FIXATION_DAT,encoding = "ISO-8859-1")


SAVE_PATH_MODEL = ''
IMAGE_DIRECTORY = ''
SAVE_DATA_PATH = ''

#set the fovea size
fovea= 30
#filter: get only the fixation part inside the maximum fovea possible
all_data = all_data_uncleared.loc[
                      (all_data["fixposx"] >= fovea) &
                      (all_data["fixposx"] <= 1024 - fovea) &
                      (all_data["fixposy"] >= fovea) &
                      (all_data["fixposy"] <= 768 - fovea) &
                       all_data["fixinvalid"] == 0
                      ]

