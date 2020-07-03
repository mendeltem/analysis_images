#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:16:59 2019

@author: mendel
"""

import pandas as pd
import numpy as np
import os

import re


def get_file_paths(start_dir, extensions = ['file']):
    """Returns all image paths with the given extensions in the directory.

    Arguments:
        start_dir: directory the search starts from.

        extensions: extensions of image file to be recognized.

    Returns:
        a sorted list of all image paths starting from the root of the file
        system.
    """
    
    if start_dir is None:
        start_dir = os.getcwd()
    img_paths = []
    for roots,dirs,files in os.walk(start_dir):
        for name in files:
          if '.' == name[0]:
            name = name[1:]
          if '_' == name[0]:
            name = name[1:]
          for e in extensions:
              if name.endswith('.' + e):

                    img_paths.append(roots + '/' + name)
    img_paths.sort()
    return img_paths
  
  
def type_of_image(path, save_dir="", model_type=""):    
  """

    Arguments:
           
        path:       absolut path of the given image
          
        save_dir:   absolut path of the save directory
          
        model_type: name of the model type
          
    Returns:  color:    1 == colore image, 0 == grayscale image
              filter_type:     original, high-pass, low-pass 
              expectedlocation:  1: expected 0: unexpected   
              targetpresent:     1: target is present, 0 target is not present
              save_dir_temp:     directory path to save
              pic_id:            id of the image
              match:             
         
    """
    
  picture_name = os.path.basename(path)
  
  pic_id = re.search("[0-9]{1,3}",picture_name)[0]
  
  if "search" in path:
    experiment_type = "search"
  elif "memory" in path:
    experiment_type = "memory"
  else:
    raise NameError("Experiment Type")
    
    
  if "original" in path:
    filter_type = "original"
    temp_filter = 0
    filtered = 0
  elif "per_hp" in path:
    filter_type = "per_hp"
    temp_filter = 1
  elif "per_lp" in path:
    filter_type = "per_lp"
    temp_filter = 1
  elif "zen_hp" in path:
    filter_type = "zen_hp"
    temp_filter = 1
  elif "zen_lp" in path:
    filter_type = "zen_lp"
    temp_filter = 1
  else:
    raise NameError("At Filter Type!")
    
  if temp_filter:  
    if "filtered" in path:
      filtered = 1
    elif "normal" in path:
      filtered = 0
    else:
      raise NameError("At filter!")
    
    
  #print("exp_type " , experiment_type)  
  
  #save_dir_temp = save_dir +"/"+model_type
  
  if model_type =="":
    save_dir_temp = save_dir +"/"  + experiment_type
  else:  
    save_dir_temp = save_dir +"/"+model_type +"/"  + experiment_type
  
  #print("pic_name:", picture_name)
 
  
  if experiment_type == "search":
    match = re.search("[ELU]", picture_name)[0]
    if match == "E":
      expectedlocation = 1
      targetpresent    = 1
    elif match == "U":
      expectedlocation = 0
      targetpresent    = 1
    elif match  == "L":
      expectedlocation = 0
      targetpresent    = 0
    else:
      raise NameError("Match!")
    match = "_"+match  
  else:
    expectedlocation = "Nan"
    targetpresent    = "NAn"
    match = ""
       
  if "color" in path:
    color = 1
    save_dir_temp = save_dir_temp + "/color"
  elif "grayscale" in path: 
    color = 0
    save_dir_temp = save_dir_temp + "/grayscale"
  else:
    raise NameError("Color Type!")
    
  #print("expectedlocation: ", expectedlocation)
  #print("targetpresent: ", targetpresent)
  
  #print("color: ", color)   
  #print("save_dir_temp: ", save_dir_temp)    
  
  return color,filter_type,filtered, experiment_type, expectedlocation, \
          targetpresent, save_dir_temp, pic_id, match


model_type = "inception"


data_1_dir_h5 = '/mnt/data/DATA/dataset/FEATURES/saved_2/' + model_type

paths = get_file_paths(data_1_dir_h5) 

#
#for j, selected_path in enumerate(paths):
#    
#  print(str(j)+" "+selected_path)
#  
#  color,filter_type,filtered, experiment_type, \
#  expectedlocation, targetpresent, \
#  save_dir_temp, pic_id, match = type_of_image(selected_path,save_dir="",
#                                               model_type=model_type)
#  
  
All_df = pd.DataFrame()

for j, selected_path in enumerate(paths):
    
  print(str(j)+" "+selected_path)

  color,filter_type,filtered, experiment_type, \
  expectedlocation, targetpresent, \
  save_dir_temp, pic_id, match = type_of_image(selected_path,save_dir="",
                                               model_type=model_type)


  pandas_data_df = pd.read_feather(selected_path)


  cols = pandas_data_df.columns


  indezes = []
  
  lis = []
  
  for i in range(1,12):
    lis.append(str(i)+"_1")
  
  for i,col in enumerate(cols):
    if  [z for z in lis if z == col] :
      #print("Index: ", i)
      indezes.append(i)
      if type(cols[i-1]) == int:
        c = cols[i-1] + 1
        #print(c)
        
  m_df = pandas_data_df.groupby('subject').mean()
  
  grouped_df = pandas_data_df.groupby('subject').mean()
  
  Mean_df = pd.DataFrame()  
  
  for r in range(grouped_df.shape[0]):
    mean_list = [] 
    for index, indi in enumerate(indezes):
      
      if len(indezes) == index+1:
        mean = grouped_df.iloc[r, indezes[index]:-1].mean()  
      else:
        mean = grouped_df.iloc[r, indezes[index]:indezes[index+1]].mean()  
         
      mean_list.append(mean)  
      
    row = np.concatenate([[grouped_df.index[r],
                           experiment_type,
                           filter_type,
                           filtered,
                           expectedlocation, 
                           targetpresent,
                           int(pic_id) ],
                  mean_list])  
      
    Mean_df = pd.concat([Mean_df, pd.Series(row).transpose()  ], axis=1)
    
  
  Mean_df = Mean_df.transpose()
  
  Mean_df = Mean_df.rename(columns = {0:"Subject",
                            1:"Experimenttype",
                            2:"Filtertype",
                            3:"Filtered",
                            4:"expectedlocation",
                            5:"targetpresent",
                            6:"Image_ID",
                            7: "Mean_of_Layer_1",
                            8: "Mean_of_Layer_2",
                            9: "Mean_of_Layer_3",
                            10: "Mean_of_Layer_4",
                            11: "Mean_of_Layer_5",
                            12: "Mean_of_Layer_6",
                            13: "Mean_of_Layer_7",
                            14: "Mean_of_Layer_8",
                            15: "Mean_of_Layer_9",
                            16: "Mean_of_Layer_10",
                            17: "Mean_of_Layer_11"
                            })
 
  All_df = pd.concat([All_df, Mean_df], axis=0)
  
All_df = All_df.reset_index()
pd.DataFrame.to_feather(All_df,fname="/mnt/data/DATA/dataset/FEATURES/saved_2/"+model_type+"_mean.file")
