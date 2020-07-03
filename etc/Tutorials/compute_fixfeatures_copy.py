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


all_data = pd.read_table(DEFAULT_EYE_FIXATION_DAT,encoding = "ISO-8859-1")



SAVE_PATH_MODEL = ''
IMAGE_DIRECTORY = ''
SAVE_DATA_PATH = ''

#set the fovea size
fovea= 30
#filter: get only the fixation part inside the maximum fovea possible
all_data = all_data.loc[
                      (all_data["fixposx"] >= fovea) &
                      (all_data["fixposx"] <= 1024 - fovea) &
                      (all_data["fixposy"] >= fovea) &
                      (all_data["fixposy"] <= 768 - fovea)
                      ]


def get_model(load_path = None, auto_save = True):
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

def get_img_paths(start_dir, extensions = ['png']):
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


def data_selecting(data,color,masktype,maskregion,fixincalid):
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
#masktype == 0 & maskregion == 0: Kontrollbedingung
    cleaned_data = data.loc[(data["colorimages"] == color) &
                         (data["masktype"]    == masktype) &
                         (data["maskregion"]  == maskregion) &
                         (data["fixinvalid"]  == fixincalid) ,
                         :]
    return cleaned_data


def get_eyemovements(dataFrame):
    """Take a DataFrame from Eyemovement Experiment and returns the 
       fixation locations. 

    Input: Selected Experment DataFrame

    Returns: DataFrame with Eyemovements

    """
    result = dataFrame.loc[:,("imageid", "fixposx", "fixposy")]
    
    return result


#dataFrame.groupby("imageid")["fixposy"].apply(list)

model = get_model()

if IMAGE_DIRECTORY == '':
    img_paths = get_img_paths(DEFAULT_IMAGE_DIRECTORY)
else:
    img_paths = get_img_paths(IMAGE_DIRECTORY)

img_sequ = ImageSequence(paths = img_paths,
                         labels = None,
                         batch_size = 1,
                         preprocessing=preprocess_input,
                         augmentation=[])


                                          
Result_df = pd.DataFrame()         
Output_df = pd.DataFrame()

Layers = np.array([])

#for i,image in enumerate(img_sequ): 
 #   picture_name = re.search("\w+(?:\.\w+)*$",img_paths[i])[0]
 #   print(picture_name)
#for i,image in enumerate(img_sequ): 
#    print("Loop :"+str(i))
#    #get the name of the picture
#    picture_name = re.search("\w+(?:\.\w+)*$",img_paths[i])[0]
#    pic_id = re.findall(r'\d+', picture_name)[0]
#
#    color_type = 0
#    pass_filter_type = 0
#    masktype = 0
#    maskregion = 0
#
#    if "color" in img_paths[i]:
#        color = 1
#        color_type = "color"
#        #print("color")
#    elif "grayscale" in img_paths[i]: 
#        color = 0
#        color_type = "grayscale"
#    
#    if "session1" in img_paths[i]:
#        session = 1
#    elif "session2": 
#        session = 2
#
#    if "high-pass" in img_paths[i]: 
#        masktype   = 2
#        maskregion = 1
#        pass_filter_type = "high-pass"
#    elif "low-pass" in img_paths[i]: 
#        masktype   = 1
#        maskregion = 1
#        pass_filter_type = "low-pass"
#    elif "original" in img_paths[i]: 
#        masktype = 0
#        maskregion = 0
#        pass_filter_type = "original"
#        
#    exp_control_color = data_selecting(  all_data,
#                                         color,
#                                         masktype,
#                                         maskregion,
#                                         fixincalid = 0)
#    
#    
#    eyemovements_2 =  exp_control_color.loc[exp_control_color["imageid"] == int(pic_id),: ]
#    
#    Output_df = Output_df.append(eyemovements_2)
    
    
    
    

    
for i,image in enumerate(img_sequ): 
    #iter_ = img_sequ.__iter__()
    #image = iter_.__next__()
    #i = 0
    print("Loop :"+str(i))
    #if i >= len(img_sequ):break
    #t_start_model_predict = time()
    p = model.predict(image, batch_size = 1)
    #t_stop_model_predict = time()
    
    #get the name of the picture
    picture_name = re.search("\w+(?:\.\w+)*$",img_paths[i])[0]
    
    #picture_name = os.path.basename(img_paths[i])
    
    #print(picture_name)
    #get the picture id
    #pic_id = re.search("\d+",picture_name)[0]
    
    pic_id = re.findall(r'\d+', picture_name)[0]
    #print(pic_id)
    #get the picture id
    #re.search("hp|lp",picture_name)
    
    #img_paths[1]
    #delete previous layer
    color_type = 0
    pass_filter_type = 0
    masktype = 0
    maskregion = 0
    
    #print(img_paths[i])
    
    #Filter für die Experimentdaten
    #to get the experiment data we take the path files name 
    if "color" in img_paths[i]:
        color = 1
        color_type = "color"
        #print("color")
    elif "grayscale" in img_paths[i]: 
        color = 0
        color_type = "grayscale"
        #print("greyscale")
    
    
    if DEFAULT_EXPERIMENT in img_paths[i]:
        session = 1
        #print(DEFAULT_EXPERIMENT)
    else: 
        session = 0
        #print("else")
        
        
    if "high-pass" in img_paths[i]: 
        masktype   = 2
        maskregion = 1
        pass_filter_type = "high-pass"
        #print("high-pass")
    elif "low-pass" in img_paths[i]: 
        masktype   = 1
        maskregion = 1
        pass_filter_type = "low-pass"
        #print("low-pass")
    elif "original" in img_paths[i]: 
        masktype = 0
        maskregion = 0
        pass_filter_type = "original"
        #print("original")
        
    
#    print("color:       ", color,      "\n",
#          "session:     ", session,    "\n",
#          "masktype:    ", masktype,   "\n",
#          "maskregion:  ", maskregion, "\n",
#          "pass_filter: ", pass_filter_type, "\n"
#          )
    
    exp_control_color = data_selecting(  all_data,
                                         color,
                                         masktype,
                                         maskregion,
                                         fixincalid = 0)
    
    
    eyemovements_2 =  exp_control_color.loc[exp_control_color["imageid"] == int(pic_id) ,: ]
    
    
    
    #Orignal Picture Size
    img_h, img_w = image.shape[1], image.shape[2]
       
    
    Layer_df = pd.DataFrame()
    if(not eyemovements_2.empty and session ==1):

        #print("entering layer loop")
        list_of_activations_SR = pd.DataFrame()
        for layer in range(len(p)):
            #inside a channel
            #layershape
            part_h   = p[layer][0].shape[0]
            part_w   = p[layer][0].shape[1]
            #number of channels
            channels = p[layer][0].shape[2]
            #scale factors for the particular feature
            scale_h = img_h / part_h
            scale_w = img_w / part_w
            #scaled fovea
            scaled_fovea_y = round(fovea / scale_h)
            scaled_fovea_x = round(fovea / scale_w)
            #the list where fixation for each channel will be saved
            activations = []

            #get the activations from each channel with eye movements
            fix_sum = []
            
#            scaled_fix_x = (round(eyemovements2["fixposx"] / scale_w)).astype(int)
#            scaled_fix_y = (round(eyemovements2["fixposy"] / scale_h)).astype(int)
            scaled_fix_x = (eyemovements_2["fixposx"] / scale_w).astype(int)
            scaled_fix_y = (eyemovements_2["fixposy"] / scale_h).astype(int)
            
            scaled_fix_y0 = scaled_fix_y - scaled_fovea_y
            scaled_fix_y1 = scaled_fix_y + scaled_fovea_y + 1
            scaled_fix_x0 = scaled_fix_x - scaled_fovea_x
            scaled_fix_x1 = scaled_fix_x + scaled_fovea_x + 1
            
            #define np
            fix_activations = np.array(np.zeros(shape=(eyemovements_2.shape[0],p[layer][0].shape[2])))
            
            
            eyemovements_2.shape
            #get the activations from each layer
            for fix in range(eyemovements_2.shape[0]):
                    fix_activations[fix,:] = p[layer][0][ 
                                    scaled_fix_y0.iloc[fix]:scaled_fix_y1.iloc[fix],
                                    scaled_fix_x0.iloc[fix]:scaled_fix_x1.iloc[fix], 
                                    :].mean(axis=(0,1))
        

                #put all the fixations to one row togher

            #save the activations in Dataframe
            list_of_activations_SR = pd.concat([list_of_activations_SR,
                                                    pd.DataFrame(fix_activations)], 
                                                    axis=1)

        
        eyemovements_2 = eyemovements_2.reset_index()
        list_of_activations_SR = list_of_activations_SR.reset_index()
        
        
        
        
        
        #print("info Shape ",eyemovements_2.shape)
        #print("list shape ",list_of_activations_SR.shape)
        
        Result_df = pd.concat([eyemovements_2, list_of_activations_SR], 
                              axis=1,
                              ignore_index=False)  
        
        #print("ResultShape: ",Result_df.shape)
        
        Output_df = Output_df.append(Result_df)
        
        if(i % 100 == 0 and i > 0):
            print("saved last piece_0")
            Output_df.to_pickle("saved/Output_"+str(i)+".h5")
            Output_df = pd.DataFrame()
            
        if i+1 >= len(img_sequ):
            print("saved last piece_1")
            Output_df.to_pickle("saved/Output_"+str(i)+".h5")  
            
        
        if i >= len(img_sequ)-1:
            print("saved last piece_2")
            Output_df.to_pickle("saved/Output_"+str(i)+".h5")
    else:
        pass
        print("Leer")
        
        

#pickle.dump( total, open( "saved/save_all_2.p", "wb" ) )



#gc.disable()
#Loaded_Result_full = pd.read_pickle("Output_full_temp.h5")



Output_df.to_pickle("saved/Output_"+str(i)+".h5")

#Loaded_Result = pd.read_pickle("saved/Output_557.h5")

Loaded_Result_1 = pd.read_pickle("saved/Output_100.h5")
Loaded_Result_2 = pd.read_pickle("saved/Output_200.h5")
Loaded_Result_3 = pd.read_pickle("saved/Output_300.h5")
Loaded_Result_4 = pd.read_pickle("saved/Output_400.h5")
Loaded_Result_5 = pd.read_pickle("saved/Output_500.h5")
Loaded_Result_6 = pd.read_pickle("saved/Output_557.h5")


Result_loaded = pd.DataFrame()

Result_loaded = Result_loaded.append(Loaded_Result_1)
Result_loaded = Result_loaded.append(Loaded_Result_2)
Result_loaded = Result_loaded.append(Loaded_Result_3)
Result_loaded = Result_loaded.append(Loaded_Result_4)
Result_loaded = Result_loaded.append(Loaded_Result_5)
Result_loaded = Result_loaded.append(Loaded_Result_6)

#Loaded_Result_full_name = pd.read_pickle("SFC_memory_foveal_features.h5")
#Loaded_Result_2 = pd.read_pickle("Output2.h5")
#Loaded_Result_3 = pd.read_pickle("Output_3.h5")
#Loaded_Result_4 = pd.read_pickle("Output_4.h5")

#Loaded_Result1 = pd.read_pickle("Output1.h5")
#Loaded_Result2 = pd.read_pickle("Output2.h5")
#aved_activation_list = pickle.load( open( "save_all.p", "rb" ) )
#t_stop_get_loading = time()
#gc.enable()

#len(saved_activation_list[1,0])
#oaded = pd.read_pickle("save_all.p")

#loaded.head()


#print ('the function model_predict takes %f' %(t_stop_model_predict-t_start_model_predict))
#print ('the function getlayer takes %f' %(t_stop_get_activation_eyemovements-t_start_get_activation_eyemovements))
#print ('the function getactivation from eyefixation takes %f' %(t_stop_get_all_layers - t_start_get_all_layers))
#print ('the function loading filtakes %f' %(t_stop_get_loading - t_start_get_loading))
#print ('the function concat layer %f' %(t_stop_concat_layer - t_start_concat_layer))

#sys.getsizeof(Output_df)

#Output_df.to_pickle("Output_full.h5")


#Output_np=np.array(Output_df)




#python has a problem with pickle.dump and very large files
def write(data, file_path):
    """Writes data (using pickle) to a file in smaller byte chunks.

    Arguments:
        data: the data to be pickled

        file_path: the file to be written to (or created)
    """
    max_bytes = 2**10 - 1
    bytes_out = pickle.dumps(data, protocol=4)
    
    
    #pickle.dump(d, open("file", 'w'), protocol=4)

    with open(file_path, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])
            
#if SAVE_DATA_PATH == '':
#    write(Output_df, DEFAULT_SAVE_DATA_PATH)
#else:
#    write(Output_df, SAVE_DATA_PATH)
#    
    
    
#def load(file_path):
#    values = np.load(file_path, mmap_mode='r')
#    with open(file_path) as f:
#        numpy.lib.format.read_magic(f)
#        numpy.lib.format.read_array_header_1_0(f)
#        f.seek(values.dtype.alignment*values.size, 1)
#        meta = pickle.loads(f.readline().decode('string_escapeframe = pd.DataFrame(values, index=meta[0], columns=meta[1])    
#    return meta