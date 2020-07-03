#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 20:55:58 2019

@author: mut, dd

it combines the Dataframe from the Experiment with the
Features from the Saved Features from the Model and save it in to
a csv datatable.

"""

import sys
import pickle
import keras as ks
import tensorflow as tf
from keras.utils import to_categorical
from keras import models
from keras import layers
import os
#import exp_library as ex
import pandas as pd
import numpy as np

from modules.library import get_img_paths,get_feature_paths,memory_usage,get_csv_paths
from modules.library import getsize

from time import time
from keras.applications.inception_v3 import preprocess_input
from modules.generators import ImageSequence

import json

from os import listdir
from os.path import isfile, join

#fevea ist 1 Grad
fovea = 15


with open("../input_file.json") as data_file:
    config = json.load(data_file)
    
    
DEFAULT_EYE_FIXATION_DAT    =  config["MEMORY"]
DEFAULT_IMAGE_DIRECTORY     = 'mem_images/'



DEFAULT_IMAGE_FEATURES_DIRECTORY = "/mnt/data/DATA/dataset/FEATURES/saved_features_from_pictures/inception/"

#feature_data_dir ="/mnt/data/mendel_moved/data/saved_features_from_pictures/color_lowpass.pickle"

#experiment data tables
MEMORY                  = config["MEMORY"]
SEARCH                  = config["SEARCH"]

#input images paths for different imagetypes
MEMORY_COLOR            = config["MEMORY_COLOR"]
MEMORY_GRAYSCALE        = config["MEMORY_GRAYSCALE"]
SEARCH_COLOR            = config["SEARCH_COLOR"]
SEARCH_GRAYSCALE        = config["SEARCH_GRAYSCALE"]



IMAGE_DIRECTORY = ""
IMAGE_FEATURES_DIRECTORY = ""

all_data_uncleared = pd.read_table(
        DEFAULT_EYE_FIXATION_DAT,encoding = "ISO-8859-1")

all_data = all_data_uncleared.loc[
                      (all_data_uncleared["fixposx"] >= fovea) &
                      (all_data_uncleared["fixposx"] <= 1024 - fovea) &
                      (all_data_uncleared["fixposy"] >= fovea) &
                      (all_data_uncleared["fixposy"] <= 768 - fovea) &
                       (all_data_uncleared["fixinvalid"] != 1) & 
                       (all_data_uncleared["sacinvalid"] != 1)
                      ]


data = pd.DataFrame()



onlyfiles = [f for f in listdir(DEFAULT_IMAGE_FEATURES_DIRECTORY) if isfile(join(DEFAULT_IMAGE_FEATURES_DIRECTORY, f))]


loaded_features = []



for i,file in enumerate(onlyfiles):
    loaded_features = pd.read_pickle(DEFAULT_IMAGE_FEATURES_DIRECTORY+file)
    


    for image_index in range(len(loaded_features)):
        print("image index: ", image_index)
        
        pic_id   = loaded_features[image_index]["pic_id"]
        pic_type = loaded_features[image_index]["type"]
        image_shape = loaded_features[image_index]["shape"]
        
        print("file name",file)
        print("pic id ", pic_id)
        
        if loaded_features[image_index]["type"] == "color_original":
            #masktype == 0 & maskregion == 0: Kontrollbedingung
            print("pic_type", pic_type)
            selected_data = all_data.loc[ 
            (all_data["imageid"] == int(pic_id))&
            ((all_data["colorimages"] == 1) )&
            ((all_data["masktype"] == 0) | (all_data["masktype"] == 1) 
            | (all_data["masktype"] == 2) ) &
            ((all_data["maskregion"]  == 0 ) | (all_data["maskregion"]  == 1 ) 
            | (all_data["maskregion"]  == 2 ) )
            ,:] 
        
        
        elif loaded_features[image_index]["type"] == "color_high-pass":
            #masktype == 0 & maskregion == 0: Kontrollbedingung
            #masktype == 2 & maskregion == 1: peripherer Hochpassfilter
            #masktype == 2 & maskregion == 2: zentraler Hochpassfilter
            print("pic_type", pic_type)
            selected_data = all_data.loc[
            (all_data["imageid"] == int(pic_id))&
            ((all_data["colorimages"] == 1) )&
            ((all_data["masktype"] == 0) | (all_data["masktype"] == 2) ) &
            ((all_data["maskregion"]  == 0 ) | (all_data["maskregion"]  == 1 ) 
            | (all_data["maskregion"]  == 2 ) )
            ,:] 
        
        elif loaded_features[image_index]["type"] == "color_low-pass":
            #masktype == 0 & maskregion == 0: Kontrollbedingung
            #masktype == 1 & maskregion == 1: peripherer Tiefpassfilter
            #masktype == 1 & maskregion == 2: zentraler Tiefpassfilter
            print("pic_type", pic_type)
            selected_data = all_data.loc[ 
            (all_data["imageid"] == int(pic_id))&
            ((all_data["colorimages"] == 1) | (all_data["colorimages"] == 1) )&
            ((all_data["masktype"] == 0) | (all_data["masktype"] == 1)  ) &
            ((all_data["maskregion"]  == 0 ) | (all_data["maskregion"]  == 1 ) 
            | (all_data["maskregion"]  == 2 ) )
            ,:] 
        
        elif loaded_features[image_index]["type"] == "greyscale_original":
            #masktype == 0 & maskregion == 0: Kontrollbedingung
            print("pic_type", pic_type)
            selected_data = all_data.loc[ (
            all_data["imageid"] == int(pic_id))&
            ((all_data["colorimages"] == 0) )&
            ((all_data["masktype"] == 0) | (all_data["masktype"] == 1)
            | (all_data["masktype"] == 2) ) &
            ((all_data["maskregion"]  == 0 ) | (all_data["maskregion"]  == 1 )
            | (all_data["maskregion"]  == 2 ) )
            ,:] 
            
        elif loaded_features[image_index]["type"] == "greyscale_high-pass":
            #masktype == 0 & maskregion == 0: Kontrollbedingung
            #masktype == 2 & maskregion == 1: peripherer Hochpassfilter
            #masktype == 2 & maskregion == 2: zentraler Hochpassfilter
            print("pic_type", pic_type)
            selected_data = all_data.loc[ (
            all_data["imageid"] == int(pic_id))&
            ((all_data["colorimages"] == 1) | (all_data["colorimages"] == 1) )&
            ((all_data["masktype"] == 0) |  (all_data["masktype"] == 2) ) &
            ((all_data["maskregion"]  == 0 ) | (all_data["maskregion"]  == 1 ) 
            | (all_data["maskregion"]  == 2 ) )
            ,:] 
        
        elif loaded_features[image_index]["type"] == "greyscale_low-pass":
            #masktype == 0 & maskregion == 0: Kontrollbedingung
            #masktype == 1 & maskregion == 1: peripherer Tiefpassfilter
            #masktype == 1 & maskregion == 2: zentraler Tiefpassfilter
            print("pic_type", pic_type)
            selected_data = all_data.loc[ (
            all_data["imageid"] == int(pic_id))&
            ((all_data["colorimages"] == 0) )&
            ((all_data["masktype"] == 0) | (all_data["masktype"] == 1)
            | (all_data["masktype"] == 2) ) &
            ((all_data["maskregion"]  == 1 ) | (all_data["maskregion"]  == 0 )
            | (all_data["maskregion"]  == 2 ) )
            ,:] 
        
        
        img_h, img_w =  image_shape[0], image_shape[1]
               
        all_layers = pd.DataFrame()
        
        #get the layer activation and ad it with axis =1  example 256 + 256 +...+ 2048
        for layer in range(len(loaded_features[image_index]['features'])):
            
            #layershape
            part_h   = loaded_features[image_index]['features'][layer][0].shape[0]
            part_w   = loaded_features[image_index]['features'][layer][0].shape[1]
            
            #scale factors for the particular feature
            scale_h = img_h / part_h
            scale_w = img_w / part_w
            
            
            #auf feature skalierte fovea
            scaled_fovea_y = round(fovea / scale_h)
            scaled_fovea_x = round(fovea / scale_w)
            
            #auf feature skalierte Fixation
            scaled_fix_x = (selected_data["fixposx"] / scale_w).astype(int)
            scaled_fix_y = (selected_data["fixposy"] / scale_h).astype(int)
            
            
            
            #Bereiche der skalierten Fovea, die vom Features extrahiert werden
            scaled_fix_y0 = scaled_fix_y - scaled_fovea_y
            scaled_fix_y1 = scaled_fix_y + scaled_fovea_y + 1
            scaled_fix_x0 = scaled_fix_x - scaled_fovea_x
            scaled_fix_x1 = scaled_fix_x + scaled_fovea_x + 1
            
            
            
            #define np, leere list von array als Platzhalter für die Features
            fix_activations = np.array(
                np.zeros(
                    shape=(
                    selected_data.shape[0],
                    loaded_features[image_index]['features'][layer][0].shape[2])))
            
            loaded_features[image_index]['features'][0].shape
                   
            ##selected_data.shape
            #get the activations from each layer
            for fix in range(selected_data.shape[0]):
                fix_activations[fix,:] = loaded_features[
                                          image_index]['features'][layer][0][ 
                                scaled_fix_y0.iloc[fix]:scaled_fix_y1.iloc[fix],
                                scaled_fix_x0.iloc[fix]:scaled_fix_x1.iloc[fix], 
                                :].mean(axis=(0,1))
                            
                            
            
            #save the activations in Dataframe
            #jede Layer wird auf axis 1 zusätzlich geadded
            all_layers = pd.concat([all_layers,
                              pd.DataFrame(fix_activations)], 
                              axis=1)
        #debug
        selected_data.shape
        all_layers.shape
        
        
        #um die die Daten zu konkatinieren muss die Index geresetet werden
        selected_data = selected_data.reset_index()
        all_layers = all_layers.reset_index()
        
        all_layers_16 = all_layers.astype("float16")
        all_layers_16["index"] = all_layers_16.index.astype("uint8")

        
        #the selected experimental data is concatinated with features
        data_16 = pd.concat([selected_data, all_layers_16], 
                                          axis=1,
                                          ignore_index=False) 
        
             
        print('16 Memory used:', memory_usage(data_16), 'Mb')
        
        #data_16.to_pickle("saved/"+str(pic_type)+"/Output_"+str(pic_type)+"_picture_"+str(pic_id)+".h5")
        data_16.to_csv("saved/"+str(pic_type)+"/Output_"+str(pic_type)+"_picture_"+str(pic_id)+".csv")
        


#Loaded_Result_1 = pd.read_pickle("saved/"+str(pic_type)+"/Output_"+str(pic_type)+"_picture_"+str(pic_id)+".h5")

 
 
#Überprüft wie viele grow die Spalte ist
#for i in range(50):
#    Loaded_Result = pd.read_pickle("saved/Output_"+str(i+1)+".h5")
#    print(i," ",data.shape)
#    shape = Loaded_Result.shape[0]
#    summe = summe + shape
#print(summe)
#    

