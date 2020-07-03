#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:12:52 2019

@author: mendel
"""

import os
from modules.m_preprocess import Unet_Data_Generator, plot_history, create_dir

img_base_width      = 1024
img_base_height     = 768

epochs = 2

data_set            = "/mnt/data/DATA/dataset/UNET/prepared_dataset/"

#Get every subdirectorys from dataset-directory
dir_names = next(os.walk(data_set))[1]


loss_list = ['kl_divergence',  
             'mean_squared_error' , 
             'binary_crossentropy' , 
             'kl_and_bce_2_outputs']


for n, directory_name in enumerate(dir_names):
  
  #if n > 2: break
  
  save_dir            = '/mnt/data/DATA/dataset/UNET/Output'
  print(str(n) + "   " + str(directory_name ) )
  
  train_data_dir = data_set+dir_names[int(n)]
  #
  save_dir  += "/"+ dir_names[int(n)]
  
  print("main_data_dir:" , train_data_dir)
   
  for downsize in [8,4]:
    model_save_name = dir_names[int(n)] + "_" +str(downsize)
      
      
    for loss in loss_list:
      
      model_save_name = dir_names[int(n)] + "_" +str(downsize) + "_"+str(loss)
      
      
      width  = int(img_base_width / downsize)
      height = int(img_base_height / downsize)
      
      query = ""
      
      query += " --img_height "+str(height)
      query += " --img_width "+str(width)
      query += " --data_set "+str(train_data_dir)
      query += " --save_dir "+str(save_dir)
      query += " --model_save_name "+str(model_save_name)
      query += " --epochs "+str(epochs)
      
      query += " --loss "+str(loss)
       
      os.system("python unet_eye_predictor.py "+str(query))



#
#
#
#data_set            = "/mnt/data/DATA/dataset/UNET/prepared_dataset/"
#
#dataset_name = dir_names[3]
#
#
#save_dir            = '/mnt/data/DATA/dataset/UNET/Output'
#print(str(n) + "   " + str(dataset_name ) )
#
#train_data_dir = data_set+dataset_name
##
#save_dir  += "/"+dataset_name
#
#create_dir(save_dir)
#
#print("main_data_dir:" , train_data_dir)
# 
##for downsize in [16,8,4]:
#  
#downsize = 4
#  
#model_save_name = dataset_name+ "_" +str(downsize)
#
#width  = int(img_base_width / downsize)
#height = int(img_base_height / downsize)
#
#query = ""
#
#query += " --img_height "+str(height)
#query += " --img_width "+str(width)
#query += " --data_set "+str(train_data_dir)
#query += " --save_dir "+str(save_dir)
#query += " --model_save_name "+str(model_save_name)
#query += " --epochs "+str(epochs)
# 
#os.system("python unet_eye_predictor_gamma.py "+str(query))
