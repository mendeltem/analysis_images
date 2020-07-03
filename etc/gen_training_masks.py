#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:31:26 2019

@author: MUT

Description create the the Training and Validation Dataset with directorys
for the Model It creates from Images and Experimental eye movement
data KDE masks which is used as a target for the model.


"""
import os
##set the working directory
#ML_PATH ="/home/mendel/Documents/work/mendel_exchange"
#
#os.chdir(ML_PATH)

from modules.library import get_img_paths,get_dat_paths
from modules.m_preprocess import copyfile
import pandas as pd


from scipy import misc
import numpy as np

import os
import re
import seaborn as sns
import subprocess

import matplotlib.pyplot as plt



#bilderdaten
dir_images = os.getcwd()+'/images'
#blickdaten
dir_fixations = os.getcwd()+'/blickdaten'
#output
output_path = os.getcwd()+"/dataset"

os.makedirs(output_path, exist_ok=True)

#monitor
my_dpi = 90
#path length for ploting kde images
pad_length = 27
#widh and height
#w = 1024+pad_length
#h = 768+pad_length
w = 1024
h = 768

#train validation split ratio
train_val_split = 0.1
#bandwith parameter for the KDE Masks
bw =30


dir_fixations_search = os.getcwd()+'/blickdaten/finalresulttab_funcsac_SFC_search.dat'

dir_fixations_memory = os.getcwd()+'/blickdaten/finalresulttab_funcsac_SFC_memory.dat'

def load_fixationDataMemory(dir_fixations_memory):
  """load the dat, filter non valid and selecte important columns
  """

  all_data = pd.read_table(dir_fixations_memory)

  filtered_data = all_data[(
      all_data["fixinvalid"] != 1)  &
  (all_data["sacinvalid"] != 1)
  ].loc[:,["subject","colorimages","imageid","masktype","maskregion",
  "fixposx","fixposy", "fixno"]]


  return filtered_data


def load_fixationDataSearch(dir_fixations_search):
  """load the dat, filter non valid and selecte important columns
  """

  all_data = pd.read_table(dir_fixations_search)

  filtered_data = all_data[(
      all_data["fixinvalid"] != 1)  &
  (all_data["sacinvalid"] != 1)
  ].loc[:,["subject","colorimages","imageid","masktype","maskregion",
  "fixposx","fixposy","targetpresent", "expectedlocation", "fixno"]]


  return filtered_data

def classify_data(filtered_data):
  """seperate different experiments and save it in a list

  choose the data associated to different experiment settings

    Arguments:
        data: DataFrame that get filtered
        Experiment type Filter
        colorimages: 1 oder 0 (color oder grayscale images)
        masktype: 0, 1, oder 2 (control, low-pass oder high-pass filter)
        maskregion: 0, 1 oder 2 (control, periphery oder center)

        Daraus ergibt sich entsprechend:
        masktype == 0 & maskregion == 0: Kontrollbedingung
        masktype == 1 & maskregion == 1: peripherer Tiefpassfilter
        masktype == 2 & maskregion == 1: peripherer Hochpassfilter
        masktype == 1 & maskregion == 2: zentraler Tiefpassfilter
        masktype == 2 & maskregion == 2: zentraler Hochpassfilter

    Returns:

  """
  list_of_exeperiments = []

  if filtered_data.shape[1] == 8:
    ex = "memory"

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 1) &
               (filtered_data["masktype"] == 0) &
               (filtered_data["maskregion"] == 0)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages", "fixno",
               "masktype","maskregion"
               ]],
    "original","color",
    ex ]

    )
    list_of_exeperiments.append([
    #pt,color
    filtered_data[ (filtered_data["colorimages"] == 1) &
               (filtered_data["masktype"] == 1) &
               (filtered_data["maskregion"] == 1)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion"]],
    "periphererTiefpass","color", ex ]
    )

    list_of_exeperiments.append([
    #ph,color
    filtered_data[ (filtered_data["colorimages"] == 1) &
               (filtered_data["masktype"] == 2) &
               (filtered_data["maskregion"] == 1)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion"]],
    "periphererHochpass","color" , ex ]
    )
    list_of_exeperiments.append([
    #zt,color
    filtered_data[ (filtered_data["colorimages"] == 1) &
               (filtered_data["masktype"] == 1) &
               (filtered_data["maskregion"] == 2)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion"]],
    "zentralerTiefpass","color" , ex ]
    )
    list_of_exeperiments.append([
    #zh,color
    filtered_data[ (filtered_data["colorimages"] == 1) &
               (filtered_data["masktype"] == 2) &
               (filtered_data["maskregion"] == 2)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion"]],
    "zentralerHochpass","color", ex ]
    )
    list_of_exeperiments.append([
    #controll,bw
    filtered_data[ (filtered_data["colorimages"] == 0) &
               (filtered_data["masktype"] == 0) &
               (filtered_data["maskregion"] == 0)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion"]],
    "original","grayscale", ex ]
    )
    list_of_exeperiments.append([
    #pt,grayscale
    filtered_data[ (filtered_data["colorimages"] == 0) &
               (filtered_data["masktype"] == 1) &
               (filtered_data["maskregion"] == 1)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion"]],
    "periphererTiefpass","grayscale", ex ]
    )
    list_of_exeperiments.append([
    #ph,grayscale
    filtered_data[ (filtered_data["colorimages"] == 0) &
               (filtered_data["masktype"] == 2) &
               (filtered_data["maskregion"] == 1)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion"]],
    "periphererHochpass","grayscale", ex ]
    )
    list_of_exeperiments.append([
    #zt,grayscale
    filtered_data[ (filtered_data["colorimages"] == 0) &
               (filtered_data["masktype"] == 1) &
               (filtered_data["maskregion"] == 2)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion"]],
    "zentralerTiefpass","grayscale", ex ])
    list_of_exeperiments.append([
    #zh,grayscale
    filtered_data[ (filtered_data["colorimages"] == 0) &
               (filtered_data["masktype"] == 2) &
               (filtered_data["maskregion"] == 2)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion"]],
    "zentralerHochpass","grayscale", ex ])
  elif filtered_data.shape[1] == 10:
    ex = "search"

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 1) &
               (filtered_data["masktype"] == 0) &
               (filtered_data["maskregion"] == 0) &
               (filtered_data["targetpresent"] == 0) &
               (filtered_data["expectedlocation"] == 0)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "original","color",ex,"target_not_present","non_expectedlocation" ])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 1) &
               (filtered_data["masktype"] == 0) &
               (filtered_data["maskregion"] == 0) &
               (filtered_data["targetpresent"] == 1) &
               (filtered_data["expectedlocation"] == 0)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "original","color",ex,"targetpresent","non_expectedlocation" ])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 1) &
               (filtered_data["masktype"] == 0) &
               (filtered_data["maskregion"] == 0) &
               (filtered_data["targetpresent"] == 1) &
               (filtered_data["expectedlocation"] == 1)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "original","color",ex,"targetpresent","expectedlocation" ])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 1) &
               (filtered_data["masktype"] == 1) &
               (filtered_data["maskregion"] == 1) &
               (filtered_data["targetpresent"] == 0) &
               (filtered_data["expectedlocation"] == 0)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "periphererTiefpass","color",ex,
    "target_not_present","non_expectedlocation"])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 1) &
               (filtered_data["masktype"] == 1) &
               (filtered_data["maskregion"] == 1) &
               (filtered_data["targetpresent"] == 1) &
               (filtered_data["expectedlocation"] == 0)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "periphererTiefpass","color",ex,"targetpresent","non_expectedlocation"])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 1) &
               (filtered_data["masktype"] == 1) &
               (filtered_data["maskregion"] == 1) &
               (filtered_data["targetpresent"] == 1) &
               (filtered_data["expectedlocation"] == 1)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "periphererTiefpass","color",ex,"targetpresent","expectedlocation" ])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 1) &
               (filtered_data["masktype"] == 2) &
               (filtered_data["maskregion"] == 1) &
               (filtered_data["targetpresent"] == 0) &
               (filtered_data["expectedlocation"] == 0)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "periphererHochpass","color",ex,"target_not_present",
    "non_expectedlocation"])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 1) &
               (filtered_data["masktype"] == 2) &
               (filtered_data["maskregion"] == 1) &
               (filtered_data["targetpresent"] == 1) &
               (filtered_data["expectedlocation"] == 0)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "periphererHochpass","color",ex,"targetpresent","non_expectedlocation"])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 1) &
               (filtered_data["masktype"] == 2) &
               (filtered_data["maskregion"] == 1) &
               (filtered_data["targetpresent"] == 1) &
               (filtered_data["expectedlocation"] == 1)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "periphererHochpass","color",ex,"targetpresent","expectedlocation" ])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 1) &
               (filtered_data["masktype"] == 1) &
               (filtered_data["maskregion"] == 2) &
               (filtered_data["targetpresent"] == 0) &
               (filtered_data["expectedlocation"] == 0)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "zentralerTiefpass","color",ex,
    "target_not_present","non_expectedlocation"])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 1) &
               (filtered_data["masktype"] == 1) &
               (filtered_data["maskregion"] == 2) &
               (filtered_data["targetpresent"] == 1) &
               (filtered_data["expectedlocation"] == 0)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "zentralerTiefpass","color",ex,"targetpresent",
    "non_expectedlocation"])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 1) &
               (filtered_data["masktype"] == 1) &
               (filtered_data["maskregion"] == 2) &
               (filtered_data["targetpresent"] == 1) &
               (filtered_data["expectedlocation"] == 1)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "zentralerTiefpass","color",ex,"targetpresent","expectedlocation" ])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 1) &
               (filtered_data["masktype"] == 2) &
               (filtered_data["maskregion"] == 2) &
               (filtered_data["targetpresent"] == 0) &
               (filtered_data["expectedlocation"] == 0)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "zentralerHochpass","color",ex,
    "target_not_present","non_expectedlocation"])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 1) &
               (filtered_data["masktype"] == 2) &
               (filtered_data["maskregion"] == 2) &
               (filtered_data["targetpresent"] == 1) &
               (filtered_data["expectedlocation"] == 0)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "zentralerHochpass","color",ex,
    "targetpresent","non_expectedlocation"])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 1) &
               (filtered_data["masktype"] == 2) &
               (filtered_data["maskregion"] == 2) &
               (filtered_data["targetpresent"] == 1) &
               (filtered_data["expectedlocation"] == 1)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "zentralerHochpass","color",ex,
    "targetpresent","expectedlocation"])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 0) &
               (filtered_data["masktype"] == 0) &
               (filtered_data["maskregion"] == 0) &
               (filtered_data["targetpresent"] == 0) &
               (filtered_data["expectedlocation"] == 0)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "original","grayscale",ex,
    "target_not_present","non_expectedlocation"])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 0) &
               (filtered_data["masktype"] == 0) &
               (filtered_data["maskregion"] == 0) &
               (filtered_data["targetpresent"] == 1) &
               (filtered_data["expectedlocation"] == 0)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "original","grayscale",ex,"targetpresent","non_expectedlocation"])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 0) &
               (filtered_data["masktype"] == 0) &
               (filtered_data["maskregion"] == 0) &
               (filtered_data["targetpresent"] == 1) &
               (filtered_data["expectedlocation"] == 1)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "original","grayscale",ex,"targetpresent","expectedlocation"])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 0) &
               (filtered_data["masktype"] == 1) &
               (filtered_data["maskregion"] == 1) &
               (filtered_data["targetpresent"] == 0) &
               (filtered_data["expectedlocation"] == 0)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "periphererTiefpass","grayscale",ex,
    "target_not_present","non_expectedlocation"])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 0) &
               (filtered_data["masktype"] == 1) &
               (filtered_data["maskregion"] == 1) &
               (filtered_data["targetpresent"] == 1) &
               (filtered_data["expectedlocation"] == 0)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "periphererTiefpass","grayscale",ex,
    "targetpresent","non_expectedlocation"])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 0) &
               (filtered_data["masktype"] == 1) &
               (filtered_data["maskregion"] == 1) &
               (filtered_data["targetpresent"] == 1) &
               (filtered_data["expectedlocation"] == 1)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "periphererTiefpass","grayscale",ex,
    "targetpresent","expectedlocation"])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 0) &
               (filtered_data["masktype"] == 2) &
               (filtered_data["maskregion"] == 1) &
               (filtered_data["targetpresent"] == 0) &
               (filtered_data["expectedlocation"] == 0)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "periphererHochpass","grayscale",ex,
    "target_not_present","non_expectedlocation"])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 0) &
               (filtered_data["masktype"] == 2) &
               (filtered_data["maskregion"] == 1) &
               (filtered_data["targetpresent"] == 1) &
               (filtered_data["expectedlocation"] == 0)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "periphererHochpass","grayscale",ex,
    "targetpresent","non_expectedlocation"])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 0) &
               (filtered_data["masktype"] == 2) &
               (filtered_data["maskregion"] == 1) &
               (filtered_data["targetpresent"] == 1) &
               (filtered_data["expectedlocation"] == 1)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "periphererHochpass","grayscale",ex,"targetpresent","expectedlocation"])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 0) &
               (filtered_data["masktype"] == 1) &
               (filtered_data["maskregion"] == 2) &
               (filtered_data["targetpresent"] == 0) &
               (filtered_data["expectedlocation"] == 0)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "zentralerTiefpass","grayscale",ex,"target_not_present",
    "non_expectedlocation" ])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 0) &
               (filtered_data["masktype"] == 1) &
               (filtered_data["maskregion"] == 2) &
               (filtered_data["targetpresent"] == 1) &
               (filtered_data["expectedlocation"] == 0)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "zentralerTiefpass","grayscale",ex,
    "targetpresent","non_expectedlocation" ])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 0) &
               (filtered_data["masktype"] == 1) &
               (filtered_data["maskregion"] == 2) &
               (filtered_data["targetpresent"] == 1) &
               (filtered_data["expectedlocation"] == 1)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "zentralerTiefpass","grayscale",ex,"targetpresent","expectedlocation"])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 0) &
               (filtered_data["masktype"] == 2) &
               (filtered_data["maskregion"] == 2) &
               (filtered_data["targetpresent"] == 0) &
               (filtered_data["expectedlocation"] == 0)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "zentralerHochpass","grayscale",ex,"target_not_present",
    "non_expectedlocation"
     ])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 0) &
               (filtered_data["masktype"] == 2) &
               (filtered_data["maskregion"] == 2) &
               (filtered_data["targetpresent"] == 1) &
               (filtered_data["expectedlocation"] == 0)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "zentralerHochpass","grayscale", ex,
    "targetpresent","non_expectedlocation"
    ])

    list_of_exeperiments.append([
    #controll,color
    filtered_data[ (filtered_data["colorimages"] == 0) &
               (filtered_data["masktype"] == 2) &
               (filtered_data["maskregion"] == 2) &
               (filtered_data["targetpresent"] == 1) &
               (filtered_data["expectedlocation"] == 1)
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]],
    "zentralerHochpass","grayscale",ex,"targetpresent","expectedlocation"
     ])
  return list_of_exeperiments

def get_image_objects_memory(filtered_search_data,dir_images):
  """creating objects each pictures with different experiment types


  input: filtered_data:      Dataframe of the Experients,
         dir_images:         Directory of the images


  return: objects of each image with eyemovements and pictures

  """
  #get the image paths
  image_paths = get_img_paths(dir_images)


  original_paths = [i for i in image_paths if "original" in i and "._O" not in i]
  memory_paths = [i for i in original_paths if "memory" in i]


  list_of_exp =  classify_data(filtered_memory_data)

  list_of_masks = []

  for u in range(len(list_of_exp)):

    for i in range(len(np.unique(list_of_exp[u][0]["imageid"]) )):
      if list_of_exp[u][3] == "memory":

        for z, image_path in enumerate(memory_paths):

          if  list_of_exp[u][2] in image_path:

            if os.path.basename(image_path) ==  str(i+1)+".png":
              path = image_path


      dataframe_temp = list_of_exp[u][0][list_of_exp[u][0]["imageid"] == i+1]

      if dataframe_temp.shape[0]:
        if list_of_exp[u][3] == "memory":
          list_of_masks.append(
                [list_of_exp[u][0][list_of_exp[u][0]["imageid"] == i+1],
                list_of_exp[u][1],
                list_of_exp[u][2],
                list_of_exp[u][3],
                path,
                dataframe_temp.shape[0]])

  return list_of_masks

def get_image_objects_search(filtered_search_data,dir_images):
  """creating objects each pictures with different experiment types


  input: filtered_data:      Dataframe of the Experients,
         dir_images:         Directory of the images


  return: objects of each image with eyemovements and pictures

  """
  #get the image paths
  image_paths = get_img_paths(dir_images)


  original_paths = [i for i in image_paths if "original" in i and "._O" not in i]
  search_paths = [i for i in original_paths if "search" in i]


  list_of_exp =  classify_data(filtered_search_data)

  list_of_masks = []

  for u in range(len(list_of_exp)):

    for i in range(len(np.unique(list_of_exp[u][0]["imageid"]) )):
      if list_of_exp[u][3] == "search":
        if list_of_exp[u][4] == 'target_not_present':
          match = "L_"
        else:
          if list_of_exp[u][5] == 'expectedlocation':
            match = "E_"
          else:
            match = "U_"

        path_0 = [f for f in search_paths if (i+1) == int(re.findall(r'\d+',
          os.path.basename(f))[0]) and match in f and list_of_exp[u][2] in f]

        try:
          path =path_0[0]
        except:
          pass

      dataframe_temp = list_of_exp[u][0][list_of_exp[u][0]["imageid"] == i+1]

      if dataframe_temp.shape[0]:
        if list_of_exp[u][3] == "search":

          list_of_masks.append(
                [list_of_exp[u][0][(list_of_exp[u][0]["imageid"] == i+1) ],
                list_of_exp[u][1],
                list_of_exp[u][2],
                list_of_exp[u][3],
                list_of_exp[u][4],
                list_of_exp[u][5],
                path,
                dataframe_temp.shape[0]])

  return list_of_masks


def create_images(image_objects,
                  bw=30,
                  train_val_split = 0.1):
  """Creating folder for different experiment type and also.

  creting kernel density estimaion masks for each picture

  Arguments:
    each_images: objects

    bw: bandwith

    train_val_split: splitting ratio

  Returns:
    None
  """
  sns.set_style("darkgrid", {"axes.facecolor": ".9",
                           'figure.facecolor': 'white',
                            'axes.spines.bottom': False,
                             'axes.spines.left': False,
                             'axes.spines.right': False,
                             'axes.spines.top': False,

                           })

  counter = 1
  input_images_dir_name = "input_images"
  label_images_dir_name = "masks"



  train_val_split = 0.1

  images_names_list = [int(re.search(r'\d+', os.path.basename(images[-2])).group()) for images in image_objects]
  unque_images_length = len(np.unique(images_names_list))
  splitting_imagename =  int(unque_images_length * (1- train_val_split))


  for i, image in enumerate(each_images):


    image_id = os.path.basename(image[-2])

    image_id = image[1]+"_" +image[2]+"_" +image[3]+"_" +image_id

    os.chdir(output_path)
    if not os.path.isdir(image[2] + "_" +image[3]):
      subprocess.call("mkdir " +image[2] + "_" +image[3], shell = True )

    #image_id = os.path.basename(image[-2])

    if not os.path.isdir(image[1]):
      subprocess.call("mkdir " +image[2] + "_" +image[3]+"/"+image[1], shell = True )

    if (counter) % splitter == 0:
      directory_name = "_validation"
    else:
      directory_name = "_training"

    if not os.path.isdir(image[2] + "_" +image[3]+"/"+image[1]+"/"+input_images_dir_name+directory_name):
      subprocess.call("mkdir "+image[2] + "_" +image[3]+"/"+image[1]+"/"+input_images_dir_name+ directory_name,
                      shell = True )

    copyfile(image[-2], image[2] + "_" +image[3]+"/"+image[1]+"/"+input_images_dir_name+ directory_name+"/"+image_id)

    if not os.path.isdir(image[2] + "_" +image[3]+"/"+image[1]+"/"+label_images_dir_name+ directory_name):
      subprocess.call("mkdir "+image[2] + "_" +image[3]+"/"+image[1]+"/"+label_images_dir_name+ directory_name ,
                      shell = True )

    counter = counter +1
    plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
    fig = sns.kdeplot(image[0].iloc[:,2],
                image[0].iloc[:,3],
                kernel = "gau",
                bw= 30,
                cmap = plt.cm.gray,
                shade=True,
                n_levels = 100,
                legend= False,
                cumulative= False)
    fig.axes.get_yaxis().set_visible(False)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.invert_yaxis()

    plt.tight_layout()
#    plt.savefig(image[2] + "_" +image[3]+"/"+image[1]+"/"+label_images_dir_name+directory_name +"/"+image_id,
#                dpi=my_dpi,
#                transparent='True',
#                bbox_inches='tight',
#                pad_inches = 0)

    plt.setp([fig.get_xticklines() +
          fig.get_yticklines() +
          fig.get_xgridlines() +
          fig.get_ygridlines()],antialiased=False)

    figure = fig.get_figure()
    figure.tight_layout(pad=0)
    figure.canvas.draw()



    image1 = np.fromstring(figure.canvas.tostring_rgb(),dtype=np.uint8,sep='')
    image1 = image1.reshape(figure.canvas.get_width_height()[::-1] + (3,))


    misc.imsave(image[2] + "_" +image[3]+"/"+image[1]+"/"+label_images_dir_name+directory_name +"/"+image_id, image1[:,:,0])



def create_images_in_one_folder(image_objects,
                  bw=30,
                  train_val_split = 0.1,
                  directory = "predictions"):
  """creating folder for different experiment type and also

  creting kernel density esitmaion masks for each picture

  creating the training and test in one folder

  example:
    train:
      dataset/train/images/train.png
      dataset/train/masks/train.png
    test:
      dataset/test/images/test.png
      dataset/test/images/test.png



  parameter:
   each_images     : objects
   bw              : bandwith
   train_val_split : splitting ratio
  """
  sns.set_style("darkgrid", {"axes.facecolor": ".9",
                           'figure.facecolor': 'white',
                            'axes.spines.bottom': False,
                             'axes.spines.left': False,
                             'axes.spines.right': False,
                             'axes.spines.top': False,

                           })

  images_names_list = [int(re.search(r'\d+', os.path.basename(images[-2])).group()) for images in image_objects]
  unque_images_length = len(np.unique(images_names_list))
  splitting_imagename =  int(unque_images_length * (1- train_val_split))


  subprocess.call("mkdir "+directory, shell = True )

  subprocess.call("cd "+directory, shell = True )

  for i, image in enumerate(image_objects):
    print("Image ",os.path.basename(image[-2]))
    numbers = int(re.search(r'\d+', os.path.basename(image[-2])).group())

    image_id = os.path.basename(image[-2])

    image_id = image[1]+"_" +image[2]+"_" +image[3]+"_" +image_id

    os.chdir(output_path)

    if not os.path.isdir("training"):
      subprocess.call("mkdir training", shell = True )

    if not os.path.isdir("validation"):
      subprocess.call("mkdir validation", shell = True )

    if not os.path.isdir("training/images"):
      subprocess.call("mkdir training/images", shell = True )

    if not os.path.isdir("training/masks"):
      subprocess.call("mkdir training/masks", shell = True )

    if not os.path.isdir("validation/images"):
      subprocess.call("mkdir validation/images", shell = True )

    if not os.path.isdir("validation/masks"):
      subprocess.call("mkdir validation/masks", shell = True )

    if numbers <= splitting_imagename:
      directory_name = "training"
      print("Image Training ",os.path.basename(image[-2]))
    else:
      directory_name = "validation"
      print("Image Validation ",os.path.basename(image[-2]))

    copyfile(image[-2], directory_name+"/images/"+image_id)

    plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
    fig = sns.kdeplot(image[0].iloc[:,2],
                image[0].iloc[:,3],
                kernel = "gau",
                bw= 30,
                cmap = plt.cm.gray,
                shade=True,
                n_levels = 100,
                legend= False,
                cumulative= False)
    fig.axes.get_yaxis().set_visible(False)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.invert_yaxis()

    plt.tight_layout()
    plt.setp([fig.get_xticklines() +
          fig.get_yticklines() +
          fig.get_xgridlines() +
          fig.get_ygridlines()],antialiased=False)

    figure = fig.get_figure()
    figure.tight_layout(pad=0)
    figure.canvas.draw()
    image1 = np.fromstring(figure.canvas.tostring_rgb(),dtype=np.uint8,sep='')
    image1 = image1.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    misc.imsave(directory_name+"/masks/"+image_id, image1[:,:,0])


def create_images_for_train(image_objects,
                  bw=30,
                  train_val_split = 0.1,    directory = "memory/"):
  """creating folder for different experiment type and also

  creting kernel density esitmaion masks for each picture

  for each experiment we build own train test directory

  example:
    train:
      dataset/color_memory/original/train/images/train.png
      dataset/color_memory/original/train/masks/train.png
    test:
      dataset/color_memory/original/test/images/train.png
      dataset/color_memory/original/test/masks/train.png


  parameter:
   each_images     : objects
   bw              : bandwith
   train_val_split : splitting ratio
  """
  sns.set_style("darkgrid", {"axes.facecolor": ".9",
                           'figure.facecolor': 'white',
                            'axes.spines.bottom': False,
                             'axes.spines.left': False,
                             'axes.spines.right': False,
                             'axes.spines.top': False,

                           })

  import re

  counter = 1
  input_images_dir_name = "images"
  label_images_dir_name = "masks"


  images_names_list = [int(re.search(r'\d+',
          os.path.basename(images[-2])).group()) for images in image_objects]


  unque_images_length = len(np.unique(images_names_list))
  splitting_imagename =  int(unque_images_length * (1- train_val_split))

  print("splitting_imagename ",splitting_imagename)
  for i, image in enumerate(image_objects):

    numbers = int(re.search(r'\d+', os.path.basename(image[-2])).group())

    print("number: ",numbers)

    image_id = os.path.basename(image[-2])

    image_id = image[1]+"_" +image[2]+"_" +image[3]+"_" +image_id

    os.chdir(output_path)
    if not os.path.isdir(image[2] + "_" +image[3]):
      subprocess.call("mkdir "+directory +image[2] + "_" +image[3], shell = True )

    #image_id = os.path.basename(image[-2])

    if not os.path.isdir(image[1]):
      counter = 0
      subprocess.call("mkdir "+directory +image[2] + "_" +image[3]\
                      +"/"+image[1], shell = True )

    if numbers <= splitting_imagename:
      directory_name = "train"
      print("Image Training",os.path.basename(image[-2]))
    else:
      directory_name = "test"
      print("Validation Training",os.path.basename(image[-2]))

    if not os.path.isdir(image[2] + "_" +image[3]+"/"\
                         +image[1]+"/"+directory_name+"/"\
                         +input_images_dir_name):
      subprocess.call("mkdir "+directory+image[2] + "_" +image[3]+\
                      "/"+image[1]+"/"+directory_name+"/"+input_images_dir_name,
                      shell = True )

    copyfile(image[-2], image[2] + "_" +image[3]+"/"+image[1]+"/"+\
             directory_name+"/"+input_images_dir_name+"/"+image_id)

    if not os.path.isdir(image[2] + "_" +image[3]+"/"+image[1]+"/"+\
                         directory_name+"/"+label_images_dir_name):
      subprocess.call("mkdir "+directory+image[2] + "_" +image[3]+"/"+image[1]+"/"\
                      +directory_name+"/"+label_images_dir_name,
                      shell = True )


    counter = counter +1
    plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
    fig = sns.kdeplot(image[0].iloc[:,2],
                image[0].iloc[:,3],
                kernel = "gau",
                bw= 30,
                cmap = plt.cm.gray,
                shade=True,
                n_levels = 100,
                legend= False,
                cumulative= False)
    fig.axes.get_yaxis().set_visible(False)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.invert_yaxis()

    plt.tight_layout()
    #    plt.savefig(image[2] + "_" +image[3]+"/"+image[1]+"/"+label_images_dir_name+directory_name +"/"+image_id,
    #                dpi=my_dpi,
    #                transparent='True',
    #                bbox_inches='tight',
    #                pad_inches = 0)

    plt.setp([fig.get_xticklines() +
          fig.get_yticklines() +
          fig.get_xgridlines() +
          fig.get_ygridlines()],antialiased=False)

    figure = fig.get_figure()
    figure.tight_layout(pad=0)
    figure.canvas.draw()



    image1 = np.fromstring(figure.canvas.tostring_rgb(),dtype=np.uint8,sep='')
    image1 = image1.reshape(figure.canvas.get_width_height()[::-1] + (3,))


    misc.imsave(directory+image[2] + "_" +image[3]+"/"+image[1]+"/"+directory_name +"/" +label_images_dir_name +"/"+image_id, image1[:,:,0])




#load and filter the experiment data
filtered_memory_data = load_fixationDataMemory(dir_fixations_memory)
filtered_search_data  =  load_fixationDataSearch(dir_fixations_search)
#instance objects for each image experiments
image_objects_memory = get_image_objects_memory(filtered_memory_data,dir_images)
image_objects_search = get_image_objects_search(filtered_search_data,dir_images)

#create the images and folders
create_images_in_one_folder(image_objects_memory, bw, train_val_split)
create_images_in_one_folder(image_objects_search, bw, train_val_split)


#reacte the mask and image in different folder
create_images_for_train(image_objects_memory, bw, train_val_split)


#create_images_for_train(image_objects_search, bw, train_val_split = 0.2)


#def main():
#
#load and filter the experiment data
#filtered_memory_data = load_fixationDataMemory(dir_fixations_memory)
#filtered_search_data  =  load_fixationDataSearch(dir_fixations_search)
#instance objects for each image experiments
#image_objects_memory = get_image_objects_memory(filtered_memory_data,dir_images)
#image_objects_search = get_image_objects_search(filtered_search_data,dir_images)
#create the images and the folders
#create_images_in_one_folder(image_objects_memory, bw, train_val_split)
#create_images_in_one_folder(image_objects_search, bw, train_val_split)
#
#if __name__ == '__main__':
#  main()
