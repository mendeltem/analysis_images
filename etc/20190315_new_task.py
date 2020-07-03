#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:31:26 2019

@author: MUT
"""

from modules.library import *
import pandas as pd


import cv2
import numpy as np
from tqdm import tqdm


from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt

dir_images = 'images'
dir_fixations = 'blickdaten'


#load the fixations
def load_fixations(dir_fixations):
  """load the dat, filter non valid and selecte important columns
  """
  fixation_paths = get_dat_paths(dir_fixations)
  all_data_list = [pd.read_table(fixation_path,encoding = "ISO-8859-1") 
                          for i,fixation_path in enumerate(fixation_paths) ]

  filtered_data = []
  for i in range(len(all_data_list)):
   filtered_data.append( all_data_list[i][(
        all_data_list[i]["fixinvalid"] != 1)  &
    (all_data_list[i]["sacinvalid"] != 1)  
    ].loc[:,["colorimages","imageid","masktype","maskregion",
    "fixposx","fixposy"]])
  
  return filtered_data

def seperate(filtered_data):
  """seperate different experiments and save it in a list
  """
  list_of_exeperiments = []
  
  for i in range(len(filtered_data)):
    
    if i == 0: 
      ex = "memory" 
    else: 
      ex = "search" 
      
    list_of_exeperiments.append([
    #controll,color
    filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
               (filtered_data[i]["masktype"] == 0) & 
               (filtered_data[i]["maskregion"] == 0)
               ].loc[:,["imageid","fixposx","fixposy"]],"controll_color",
    ex ]
  
    )
    list_of_exeperiments.append([
    #pt,color
    filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
               (filtered_data[i]["masktype"] == 1) & 
               (filtered_data[i]["maskregion"] == 1)
               ].loc[:,["imageid","fixposx","fixposy"]],
    "periphererTiefpass_color", ex ]
    )
  
    list_of_exeperiments.append([
    #ph,color
    filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
               (filtered_data[i]["masktype"] == 2) & 
               (filtered_data[i]["maskregion"] == 1)
               ].loc[:,["imageid","fixposx","fixposy"]],
    "periphererHochpass_color" , ex ]
    )
    list_of_exeperiments.append([
    #zt,color
    filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
               (filtered_data[i]["masktype"] == 1) & 
               (filtered_data[i]["maskregion"] == 2)
               ].loc[:,["imageid","fixposx","fixposy"]],
    "zentralerTiefpass_color" , ex ]
    )
    list_of_exeperiments.append([
    #zh,color
    filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
               (filtered_data[i]["masktype"] == 2) & 
               (filtered_data[i]["maskregion"] == 2)
               ].loc[:,["imageid","fixposx","fixposy"]],
    "zentralerHochpass_color", ex ]
    )
    list_of_exeperiments.append([
    #controll,bw
    filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
               (filtered_data[i]["masktype"] == 0) & 
               (filtered_data[i]["maskregion"] == 0)
               ].loc[:,["imageid","fixposx","fixposy"]],
    "controll_bw", ex ]
    )
    list_of_exeperiments.append([
    #pt,bw
    filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
               (filtered_data[i]["masktype"] == 1) & 
               (filtered_data[i]["maskregion"] == 1)
               ].loc[:,["imageid","fixposx","fixposy"]],
    "periphererTiefpass,bw", ex ]
    )
    list_of_exeperiments.append([
    #ph,bw
    filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
               (filtered_data[i]["masktype"] == 2) & 
               (filtered_data[i]["maskregion"] == 1)
               ].loc[:,["imageid","fixposx","fixposy"]],
    "periphererHochpass_bw", ex ]
    )
    list_of_exeperiments.append([
    #zt,bw
    filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
               (filtered_data[i]["masktype"] == 1) & 
               (filtered_data[i]["maskregion"] == 2)
               ].loc[:,["imageid","fixposx","fixposy"]],
    "zentralerTiefpass_bw", ex ])
    list_of_exeperiments.append([
    #zh,bw
    filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
               (filtered_data[i]["masktype"] == 2) & 
               (filtered_data[i]["maskregion"] == 2)
               ].loc[:,["imageid","fixposx","fixposy"]],
    "zentralerHochpass_bw", ex ])
  return list_of_exeperiments
  
  
def each_image(filtered_data):
  
  list_of_exp = seperate(filtered_data)

  list_of_masks = []
  
  for u in range(len(list_of_exp)):
  
    for i in range(len(np.unique(list_of_exp[u][0]["imageid"]) )):
      list_of_masks.append(
          [list_of_exp[u][0][list_of_exp[u][0]["imageid"] == i+1],
          list_of_exp[u][1],
          list_of_exp[u][2]]
          
          )
  return list_of_masks   





#get the image paths
image_paths = get_img_paths(dir_images)

#load the fixations
filtered_data = load_fixations(dir_fixations)

#experimente in eine Liste trennen
each_images =  each_image(filtered_data)


X = np.array([[1, 1], [2, 1], [3, 2], [1, 1], [2, 1], [3, 2],
              [4, 1], [5, 4], [6, 2], [3, 8], [9, 1], [9, 6]])
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
kde.score_samples(X)

plt.plot(X)    
    


import scipy.stats as stats
x1 = np.array([-7, -5, 1, 4, 5.])
kde = stats.gaussian_kde(x1)
xs = np.linspace(-10, 10, num=50)
y1 = kde(xs)
kde.set_bandwidth(bw_method='silverman')
y2 = kde(xs)
kde.set_bandwidth(bw_method=kde.factor / 3.)
y3 = kde(xs)

plt.plot(y3)
plt.plot(y2)
plt.plot(y1)




def main(dir_fixations, dir_images):
  """

  creates target directories

  reads fixation data
  find corresponding image to determine save name and original size
  recalculates fixations on target size Height 768 Width 512
  creates normed fixation map
  smooths fixation map (gaussian kernel)
  creates mask with values between 0 and 1
  saves masks

  reads images
  saves resized (rotated) images

  """

main(dir_fixations, dir_images)

#to create directories
#import os
#os.makedirs

