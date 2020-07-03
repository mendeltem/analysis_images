#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:06:35 2019

@author: mendel
"""

import pickle
import os
import matplotlib.pyplot as plt
from os.path import join, splitext

output_path = '/mnt/data/DATA/dataset/UNET/Output'

from pylab import rcParams
rcParams['figure.figsize'] = 25, 15


def plot_history_bce_dice_loss(history, save_dir = 0,title =""):
  for lab in history.keys():
    if "bce_dice_loss" == lab  or "val_bce_dice_loss" == lab:
      plt.plot(history[lab], label = lab +" " + str(round(history[lab][-1], 2  )))
  plt.legend()
  if save_dir:
    plt.savefig(join(save_dir,title+"_bce_dice_loss.jpg"))    
  plt.title(title)    
  plt.show()

def plot_history_mse_loss(history, save_dir = 0,title =""):
  for lab in history.keys():
    if "mean_absolute_error" == lab  or "val_mean_absolute_error" == lab:
      plt.plot(history[lab], label = lab +" " + str(round(history[lab][-1], 2  )))
  plt.legend()
  if save_dir:
    plt.savefig(join(save_dir,title+"_mse_loss.jpg"))    
  plt.title(title)    
  plt.show()


def print_and_save_history(path):
  history = pickle.load( open( path, 'rb' ) )  

  name = path.rsplit('/', 1)[-1]
  path_till = path.rsplit('/', 1)[:1][0]
  plot_history_bce_dice_loss(history,save_dir = path_till, title = name )
  plot_history_mse_loss(history,save_dir = path_till, title = name )


print_and_save_history('/mnt/data/DATA/dataset/UNET/Output/ANKES_BW_25_control/ANKES_BW_25_control_4trainHistoryDict')
print_and_save_history('/mnt/data/DATA/dataset/UNET/Output/ANKES_BW_25_control/ANKES_BW_25_control_8trainHistoryDict')
print_and_save_history('/mnt/data/DATA/dataset/UNET/Output/set_salicon_bw_30/set_salicon_bw_30_4trainHistoryDict')
print_and_save_history('/mnt/data/DATA/dataset/UNET/Output/set_salicon_bw_30/set_salicon_bw_30_8trainHistoryDict')
print_and_save_history('/mnt/data/DATA/dataset/UNET/Output/set_salicon_bw_25/set_salicon_bw_25_4trainHistoryDict')
print_and_save_history('/mnt/data/DATA/dataset/UNET/Output/set_salicon_bw_25/set_salicon_bw_25_8trainHistoryDict')
print_and_save_history('/mnt/data/DATA/dataset/UNET/Output/ANKES_BW_30_memory_control/ANKES_BW_30_memory_control_4trainHistoryDict')
print_and_save_history('/mnt/data/DATA/dataset/UNET/Output/ANKES_BW_30_memory_control/ANKES_BW_30_memory_control_8trainHistoryDict')






