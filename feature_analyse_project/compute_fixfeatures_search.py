#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 15:17:26 2018

@author: DD, UT

compute and save intermediate representations of fixations on images from the
inception_v3 net.
"""


import keras
import os

#import exp_library as ex
import pandas as pd
import numpy as np
import re
import json
import feather

foveal=0
feature = 0


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from keras.applications.inception_v3 import preprocess_input
from modules.generators import ImageSequence
from modules.library import create_dir

with open('../input_file.json') as data_file:
    config = json.load(data_file)

DEFAULT_SAVE_PATH_MODEL     = 'model_inception_v3_mixedlayeroutputs_auto.h5'

SAVE_DIR                    = '/mnt/data/DATA/dataset/FEATURES/saved_2'
DEFAULT_IMAGE_DIRECTORY     = config["SEARCH_IMAGES"]
DEFAULT_EYE_FIXATION_DAT    = config["SEARCH"]
model_type = "inception"

all_data = pd.read_table(DEFAULT_EYE_FIXATION_DAT,encoding = "ISO-8859-1")

SAVE_PATH_MODEL = ''
IMAGE_DIRECTORY = ''
SAVE_DATA_PATH = ''

#set the fovea size
fovea= 64

#filter: get only the fixation part inside the maximum fovea possible
all_data = all_data.loc[
                      (all_data["fixposx"] >= fovea) &
                      (all_data["fixposx"] <= 1024 - fovea) &
                      (all_data["fixposy"] >= fovea) &
                      (all_data["fixposy"] <= 768 - fovea)
                      ]

def get_model_inception(load_path = None, auto_save = True):
    """Loads or instantiates a model based on inception_v3 returning the
    outputs of all 11 mixed layers with indices from 0 to 10.

    Arguments:
        load_path: path of the already instantiated saved model.

        auto_save: if True, saves the model in the default save path, if model
        was not loaded.

    Returns:
        a keras model with all full mixed layers of inception_V3 as output
    """

    if load_path is None:
        if SAVE_PATH_MODEL == '':
            load_path = DEFAULT_SAVE_PATH_MODEL
        else:
            load_path = SAVE_PATH_MODEL
    try:
        model = keras.models.load_model(load_path)

    except OSError:
        inc_v3 = keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')

        def get_mixed_layer_names():
            layer_names = []
            for layer in inc_v3.layers:
                if 'mixed' in layer.name:
                    layer_names.append(layer.name)
            return layer_names

        mixed_layer_names = get_mixed_layer_names()

        main_mixed_layer_names = [ln for ln in mixed_layer_names if '_' not in ln]

        x = inc_v3.input
        outs = []
        for ln in main_mixed_layer_names:
            outs.append(inc_v3.get_layer(ln).output)
        model = keras.Model(inputs=x, outputs=outs)
        if auto_save:
            model.save(DEFAULT_SAVE_PATH_MODEL, include_optimizer=False)
    return model

def get_model_exception(load_path = None, auto_save = True):
    """Loads or instantiates a model based on inception_v3 returning the
    outputs of all 11 mixed layers with indices from 0 to 10.

    Arguments:
        load_path: path of the already instantiated saved model.

        auto_save: if True, saves the model in the default save path, if model
        was not loaded.

    Returns:
        a keras model with all full mixed layers of inception_V3 as output
    """

    if load_path is None:
        if SAVE_PATH_MODEL == '':
            load_path = DEFAULT_SAVE_PATH_MODEL
        else:
            load_path = SAVE_PATH_MODEL
    try:
        model = keras.models.load_model(load_path)

    except OSError:
        xception = keras.applications.Xception(include_top=False,
                                             weights='imagenet')

        def get_mixed_layer_names():
            layer_names = []
            for layer in xception.layers:
                if 'mixed' in layer.name:
                    layer_names.append(layer.name)
            return layer_names

        mixed_layer_names = get_mixed_layer_names()

        main_mixed_layer_names = [ln for ln in mixed_layer_names if '_' not in ln]

        x = xception.input
        outs = []
        for ln in main_mixed_layer_names:
            outs.append(xception.get_layer(ln).output)
        model = keras.Model(inputs=x, outputs=outs)
        if auto_save:
            model.save(DEFAULT_SAVE_PATH_MODEL, include_optimizer=False)
    return model


def get_img_paths(start_dir, extensions = ['png']):
    """Returns all image paths with the given extensions in the directory.

    Arguments:
        start_dir: directory the search starts from.

        extensions: extensions of image file to be recognized.

    Returns:
        a sorted list of all image paths starting from the root of the file
        system.
    """    
    extensions = ['png']
    
    
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
  

def get_data_paths(start_dir, extensions = ['csv', 'h5']):
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
            for e in extensions:
                if name.endswith('.' + e):
                    img_paths.append(roots + '/' + name)
    img_paths.sort()
    return img_paths

def data_selecting(data,imageid,color, masktype, maskregion):
    """choose the data associated to different experiment settings

    Arguments:
        data: DataFram that get filtered
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
        
        fixinvalid: 0 oder 1 (Fixation ist valide oder invalide); 
        invalide Fixationen sind 
        z.B. blinks oder Fixationen außerhalb des Monitors/Bildes
            
        group:  0:image and subjec 
                1: image 
                2:subject
        head: first indizes of the group

            ...
    Returns: DataFrame with lists of Eyemovement
    """
    if maskregion == 2:
        selected_data = data.loc[(data["colorimages"] == color) &
                                 (data["imageid"] == int(imageid))&
                                 (data["masktype"]    == masktype) &
                                 (data["maskregion"]  == maskregion) 
                         ,
                         :]
    else:
    
        selected_data = data.loc[ ((data["colorimages"] == color))&
                                 (data["imageid"] == int(imageid))&
                                ((data["masktype"] == 0) | 
                                (data["masktype"] == 1) | 
                                (data["masktype"] == 2)) &
                                ((data["maskregion"]  == 1 ) | 
                                    (data["maskregion"]  == 0 ))
                                ,
                                :] 
          
    return selected_data
  
def get_feature(p, fovea, image, selected_data): 
    """Get a mean of an area of all features in the given layers in p.
    The result is then concatenated with a selected data.

    Arguments:
        p:  all layers from the model

        fovea:  fovea size
          
        image:  image
          
        selected_data: the selected dataframe to concatenate 


    Returns:
       Result_df: feature areas concatenated with a selected dataframe
    """
    img_h, img_w = image.shape[1], image.shape[2]
    Result_df = pd.DataFrame()   
    list_of_activations_SR = pd.DataFrame()
    
    
    for layer in range(len(p)):
        #inside a channel
        #layershape
        part_h   = p[layer][0].shape[0]
        part_w   = p[layer][0].shape[1]
        #number of channels
        #scale factors for the particular feature
        scale_h = img_h / part_h
        scale_w = img_w / part_w
        #scaled fovea
        scaled_fovea_y = round(fovea / scale_h)
        scaled_fovea_x = round(fovea / scale_w)
        #the list where fixation for each channel will be saved

  
        #get the activations from each channel with eye movements
        scaled_fix_x = (selected_data["fixposx"] / scale_w).astype(int)
        scaled_fix_y = (selected_data["fixposy"] / scale_h).astype(int)
        
        scaled_fix_y0 = scaled_fix_y - scaled_fovea_y
        scaled_fix_y1 = scaled_fix_y + scaled_fovea_y + 1
        scaled_fix_x0 = scaled_fix_x - scaled_fovea_x
        scaled_fix_x1 = scaled_fix_x + scaled_fovea_x + 1
        
        #define np
        fix_activations = np.array(np.zeros(shape=(selected_data.shape[0],
                                                   p[layer][0].shape[2])))
        
        ##selected_data.shape
        #get the activations from each layer
        for fix in range(selected_data.shape[0]):
          
                foveal_area = p[layer][0][ 
                              scaled_fix_y0.iloc[fix]:scaled_fix_y1.iloc[fix],
                              scaled_fix_x0.iloc[fix]:scaled_fix_x1.iloc[fix], 
                              :]
                feature_area = p[layer][0]
                foveal_mean = foveal_area.mean(axis=(0,1)) 
                feature_mean =  p[layer][0].mean(axis=(0,1))
                
                if fix == 1 and layer == 1:
                  print("the value for the fixation ", fix,"and layer ", layer)
                  print("foveal_shape:  ", foveal_area.shape)
                  print("feature_shape: ", feature_area.shape)
                  print("foveal_mean_shape:  ", foveal_mean.shape)
                  print("feature_mean_shape: ", feature_mean.shape)    
                  
                  
                  global foveal
                  
                  global feature
                  
                  foveal=foveal_mean
                  feature =feature_mean
                  print("foveal_mean:  ", foveal)
                  print("feature_mean: ", feature)  
    
                fix_activations[fix,:] = foveal_mean - feature_mean
            #put all the fixations to one row togher
            
        temp_df  = pd.DataFrame(fix_activations)    
        
        #add layer to columnname
        temp_df.columns = [str(layer+1) + "_" +
                           str(col+1) for col in temp_df.columns]
  
        #save the activations in Dataframe
        #jede Layer wird auf axis 1 zusätzlich geadded
        list_of_activations_SR = pd.concat([list_of_activations_SR,
                                                temp_df], 
                                                axis=1)
  
    #um die die Daten zu konkatinieren muss die Index geresetet werden
    selected_data = selected_data.reset_index()
    #list_of_activations_SR = list_of_activations_SR.reset_index()
    
    #list_of_activations_SR = list_of_activations_SR
    
    #ausgabe vom Form
    #debug  
    #print("list shape ",list_of_activations_SR.shape)
    #Die Experimentdaten und Aktivierungsdaten aus dem Feature werden
    #konkatiniert
  
    Result_df = pd.concat([selected_data, list_of_activations_SR], 
                              axis=1,
                              ignore_index=False)  
    
    
    #feather.write_dataframe(Result_df, save_dir)
    
    return Result_df
              
  
  
def get_feature2(p, fovea, image, selected_data): 
    """Get a mean of an area of all features in the given layers in p.
    The result is then concatenated with a selected data.

    Arguments:
        p:  all layers from the model

        fovea:  fovea size
          
        image:  image
          
        selected_data: the selected dataframe to concatenate 


    Returns:
       Result_df: feature areas concatenated with a selected dataframe
    """
    img_h, img_w = image.shape[1], image.shape[2]
    Result_df = pd.DataFrame()   
    list_of_activations_SR = pd.DataFrame()
    
    
    for layer in range(len(p)):
        #inside a channel
        #layershape
        part_h   = p[layer][0].shape[0]
        part_w   = p[layer][0].shape[1]
        #number of channels
        #scale factors for the particular feature
        scale_h = img_h / part_h
        scale_w = img_w / part_w
        #scaled fovea
        scaled_fovea_y = round(fovea / scale_h)
        scaled_fovea_x = round(fovea / scale_w)
        #the list where fixation for each channel will be saved

  
        #get the activations from each channel with eye movements
        scaled_fix_x = (selected_data["fixposx"] / scale_w).astype(int)
        scaled_fix_y = (selected_data["fixposy"] / scale_h).astype(int)
        
        #creating smaller foveal arrea
        scaled_fix_y0 = scaled_fix_y - scaled_fovea_y
        scaled_fix_y1 = scaled_fix_y + scaled_fovea_y + 1
        scaled_fix_x0 = scaled_fix_x - scaled_fovea_x
        scaled_fix_x1 = scaled_fix_x + scaled_fovea_x + 1
        
        #define np
        fix_activations = np.array(np.zeros(shape=(selected_data.shape[0],
                                                   p[layer][0].shape[2])))
        
        ##selected_data.shape
        #get the activations from each layer
        for fix in range(selected_data.shape[0]):
                fix_activations[fix,:] = p[layer][0][ 
                              scaled_fix_y0.iloc[fix]:scaled_fix_y1.iloc[fix],
                              scaled_fix_x0.iloc[fix]:scaled_fix_x1.iloc[fix], 
                              :].mean(axis=(0,1))
    
  
            #put all the fixations to one row togher
            
        temp_df  = pd.DataFrame(fix_activations)    
        
        #add layer to columnname
        temp_df.columns = [str(layer+1) + "_" +
                           str(col+1) for col in temp_df.columns]
  
        #save the activations in Dataframe
        #jede Layer wird auf axis 1 zusätzlich geadded
        list_of_activations_SR = pd.concat([list_of_activations_SR,
                                                temp_df], 
                                                axis=1)
  
    #um die die Daten zu konkatinieren muss die Index geresetet werden
    selected_data = selected_data.reset_index()

    Result_df = pd.concat([selected_data, list_of_activations_SR], 
                              axis=1,
                              ignore_index=False)  
    
    
    return Result_df  
  
  
  
  
  
def sel_experiment_data(all_data,
                        pic_id,
                        color, 
                        expectedlocation,
                        targetpresent,
                        masktype, 
                        maskregion
                        ):
    """Select a given experiment from a dataframe

    Arguments:
        all_data: (DataFrame) dataframe
        pic_id:    (int)      image id
        color:   (int)   1 == colore image, 0 == grayscale image
        expectedlocation: (int)
        targetpresent:  (int)
        masktype :      (int)
        maskregion :    (int)



    Returns:
       selected_data: dataframe that is selected
    """
    
    
    if "expectedlocation" in all_data.columns or \
    "targetpresent" in  all_data.columns:
      #print("search")
      
      #print("expectedlocation ",expectedlocation )
      #print("targetpresent ",targetpresent )
      selected_data = all_data.loc[
           (all_data["imageid"] == int(pic_id))&
           (all_data["colorimages"] == color )&
           (all_data["masktype"] == masktype ) &
           (all_data["maskregion"]  == maskregion )&
           (all_data["expectedlocation"]  == expectedlocation )&
           (all_data["targetpresent"]  == targetpresent )
           ,:]
  
    elif "expectedlocation" not in all_data.columns or \
    "targetpresent" not in all_data.columns:
      #print("memory")
      selected_data = all_data.loc[
           (all_data["imageid"] == int(pic_id))&
           (all_data["colorimages"] == color )&
           (all_data["masktype"] == masktype ) &
           (all_data["maskregion"]  == maskregion )
           ,:]
    
    
    return selected_data
  
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
  
  save_dir_temp = save_dir +"/"+model_type

  picture_name = os.path.basename(path)
  
  pic_id = re.search("[0-9]{1,3}",picture_name)[0]
  
  if "search" in path:
    experiment_type = "search"
  elif "memory" in path:
    experiment_type = "memory"
    
  if "original" in path:
    filter_type = "original"
  elif "high-pass" in path:
    filter_type = "high-pass"
  elif "low-pass" in path:
    filter_type = "low-pass"
    
  #print("exp_type " , experiment_type)  
  
  save_dir_temp = save_dir +"/"+model_type+"/"  + experiment_type
  
  print("pic_name:", picture_name)
  
  match = re.search("[ELU]", picture_name)[0]
  
  #print(match)
  
  if match == "E":
    expectedlocation = 1
    targetpresent    = 1
  elif match == "U":
    expectedlocation = 0
    targetpresent    = 1
  elif match  == "L":
    expectedlocation = 0
    targetpresent    = 0
    
  match = "_"+match  
    
  if "color" in path:
    color = 1
    save_dir_temp = save_dir_temp + "/color"
  elif "grayscale" in path: 
    color = 0
    save_dir_temp = save_dir_temp + "/grayscale"
    
  #print("expectedlocation: ", expectedlocation)
  #print("targetpresent: ", targetpresent)
  
  #print("color: ", color)   
  #print("save_dir_temp: ", save_dir_temp)    
  
  return color,filter_type,\
         expectedlocation, targetpresent, save_dir_temp, pic_id, match
         
    
    
if model_type == "inception":              
  model = get_model_inception()
elif model_type == "exception":
  model = get_model_exception()

if IMAGE_DIRECTORY == '':
    img_paths = get_img_paths(DEFAULT_IMAGE_DIRECTORY)
else:
    img_paths = get_img_paths(IMAGE_DIRECTORY)

img_sequ = ImageSequence(paths = img_paths,
                         labels = None,
                         batch_size = 1,
                         preprocessing=preprocess_input,
                         augmentation=[])

for i,image in enumerate(img_sequ): 
    print(i)
    if i == 2:
      break
  
    print(img_paths[i])

    
    color,filter_type,\
    expectedlocation, targetpresent, \
    save_dir_temp, pic_id, match = type_of_image(img_paths[i],
                                          SAVE_DIR ,
                                          model_type)
    
    
    #print("filter_type", filter_type)
    
    #print("expectedlocation", pic_id)
    
    #print("pic_id", pic_id)
    

    
    #print("Loop :"+str(i))
    #if i >= len(img_sequ):break
    
    #predicted the features
    p = model.predict(image, batch_size = 1)
           
    if "original" == filter_type:
      
        #print("original")
        #colorimages: 1 
        #masktype == 0 & maskregion == 0: Kontrollbedingung
        #Kontrollbedingung ohne Farben
        selected_data = sel_experiment_data(all_data, 
                                            pic_id, 
                                            color, 
                                            expectedlocation,
                                            targetpresent, 
                                            masktype = 0,
                                            maskregion = 0,
                                            )
        df = get_feature(p, fovea, image, selected_data)
        
        if df.shape[0]:
          create_dir(save_dir_temp+"/original/")
          
          feather.write_dataframe(
              df,save_dir_temp+"/original/" +str(pic_id)+match+".file")
        
        #masktype == 2 & maskregion == 2: zentraler Hochpassfilter           
        selected_data = sel_experiment_data(all_data, 
                                            pic_id, 
                                            color, 
                                            expectedlocation,
                                            targetpresent, 
                                            masktype = 2,
                                            maskregion = 2,
                                            )
        if df.shape[0]:
          df = get_feature(p, fovea, image, selected_data)
          
          create_dir(save_dir_temp+"/zen_hp/original/" )
          
          feather.write_dataframe(
              df,save_dir_temp+"/zen_hp/original/" +str(pic_id)+match+".file")
        
        # masktype == 2 & maskregion == 2: zentraler Tiefpassfilter           
        selected_data = sel_experiment_data(all_data, 
                                pic_id, 
                                color, 
                                expectedlocation,
                                targetpresent, 
                                masktype = 1,
                                maskregion = 2,
                                )
        
        df = get_feature(p, fovea, image, selected_data)
        
        if df.shape[0]:
          
          create_dir(save_dir_temp+"/zen_lp/original/" )
          
          feather.write_dataframe(
              df,save_dir_temp+"/zen_lp/original/" +str(pic_id)+match+".file")
        
        
        #masktype == 2 & maskregion == 1: peripherer Hochpassfilter         
        selected_data = sel_experiment_data(all_data, 
                                pic_id, 
                                color, 
                                expectedlocation,
                                targetpresent,
                                masktype = 2,
                                maskregion = 1,
                                )
        df = get_feature(p, fovea, image, selected_data)
        
        if df.shape[0]:
        
          create_dir(save_dir_temp+"/per_hp/original/" )
          
          feather.write_dataframe(
              df,save_dir_temp+"/per_hp/original/" +str(pic_id)+match+".file")


        #masktype == 1 & maskregion == 1: peripherer Tiefpassfilter        
        selected_data = sel_experiment_data(all_data, 
                                pic_id, 
                                color, 
                                expectedlocation,
                                targetpresent, 
                                masktype = 1,
                                maskregion = 1,
                                )
        if df.shape[0]:
          df = get_feature(p, fovea, image, selected_data)
          
          create_dir(save_dir_temp+"/per_lp/original/" )
          
          feather.write_dataframe(
              df,save_dir_temp+"/per_lp/original/" +str(pic_id)+match+".file")
        


    elif "high-pass" == filter_type: 
        #print("high-pass zentral")
        #masktype == 2 & maskregion == 2: zentraler Hochpassfilter           
        selected_data = sel_experiment_data(all_data, 
                                            pic_id, 
                                            color, 
                                            expectedlocation,
                                            targetpresent,
                                            masktype = 2,
                                            maskregion = 2
                                            )
        
        df = get_feature(p, fovea, image, selected_data)
        
        #print(df.shape)
                
        if df.shape[0]:
        
          create_dir(save_dir_temp+"/zen_hp/filtered/" )
        
          feather.write_dataframe(
            df,save_dir_temp+"/zen_hp/filtered/" +str(pic_id)+match+".file")
        
        #print("high-pass peripherer")
        #masktype == 2 & maskregion == 1: peripherer Hochpassfilter         
        selected_data = sel_experiment_data(all_data, 
                                pic_id, 
                                color, 
                                expectedlocation,
                                targetpresent,
                                masktype = 2,
                                maskregion = 1
                                )
        
        df = get_feature(p, fovea, image, selected_data)
        
        #print(df.shape)
        
        if df.shape[0]:
        
          create_dir(save_dir_temp+"/per_hp/filtered/" )
          
          feather.write_dataframe(
              df,save_dir_temp+"/per_hp/filtered/" +str(pic_id)+match+".file")
        
           
    elif "low-pass" == filter_type: 
        #print("low-pass zentral")

        # masktype == 2 & maskregion == 2: zentraler Tiefpassfilter       
        selected_data = sel_experiment_data(all_data, 
                                pic_id, 
                                color,
                                expectedlocation,
                                targetpresent, 
                                masktype = 1,
                                maskregion = 2
                                )
        
        df = get_feature(p, fovea, image, selected_data)
        
        if df.shape[0]:
          create_dir(save_dir_temp+"/zen_lp/filtered/" )
          
          feather.write_dataframe(
              df,save_dir_temp+"/zen_lp/filtered/" +str(pic_id)+match+".file")
        
        #print("low-pass peripherer")
        #masktype == 1 & maskregion == 1: peripherer Tiefpassfilter        
        selected_data = sel_experiment_data(all_data, 
                                pic_id, 
                                color, 
                                expectedlocation,
                                targetpresent, 
                                masktype = 1,
                                maskregion = 1
                                )
        if df.shape[0]:
          df = get_feature(p, fovea, image, selected_data)
          
          create_dir(save_dir_temp+"/per_lp/filtered/" )
          
          feather.write_dataframe(
              df,save_dir_temp+"/per_lp/filtered/" +str(pic_id)+match+".file")
          
        