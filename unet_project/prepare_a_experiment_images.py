#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:31:26 2019

@author: MUT,DD

Create the training and validation datasets with kernel density estimation 
masks from the eye movement experiment fixations. 
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import seaborn as sns
import re 
import imageio
from skimage.transform import resize
import json
from joblib import Parallel, delayed
import warnings


from modules.library import create_dir, intersection_of_2_list
from modules.library import get_keyname_from_true_dict, get_img_paths
from modules.library import  get_word_contains_in_list,\
get_keynames_contain_keyword,\
get_first_number_from_path,create_kde,save_image,kde2D_plot 


warnings.filterwarnings("ignore", category = UserWarning)
warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)


#from modules.library import get_img_paths
with open("input_file.json") as data_file:
    config = json.load(data_file)
#Fixed Value for the Plot
#radnom seed
seed = 0
random.seed(seed)
np.random.seed(seed=seed)
"""Parameter for the process"""
#widh and height
height = 768
width = 1024

#experiment data tables
MEMORY                  = config["MEMORY"]
SEARCH                  = config["SEARCH"]

#input images paths for different imagetypes
MEMORY_COLOR            = config["MEMORY_COLOR"]
MEMORY_GRAYSCALE        = config["MEMORY_GRAYSCALE"]
SEARCH_COLOR            = config["SEARCH_COLOR"]
SEARCH_GRAYSCALE        = config["SEARCH_GRAYSCALE"]

#seaborn.pydata.org/generated/seaborn.kdeplot.html
#output dir
output_path = config["DATASET"] + "ANKES_BW_TEST/"

#Code for shape of kernel to fit with. Bivariate KDE can only use gaussian kernel.
#KERNEL : {‘gau’ | ‘cos’ | ‘biw’ | ‘epa’ | ‘tri’ | ‘triw’ }, optional
KERNEL     = "gau"
#Name of reference method to determine kernel size,
#scalar factor, or scalar for each dimension of the bivariate plot.
#Note that the underlying computational libraries have different 
#interperetations for this parameter: statsmodels uses it directly,
#but scipy treats it as a scaling factor for the standard deviation of the data.
#BANDWITH : {‘scott’ | ‘silverman’ | scalar | pair of scalars }, optional
BANDWITH   = 30

CMAP       = plt.cm.gray
#If True, shade in the area under the KDE curve 
#(or draw with filled contours when data is bivariate).
#shade : bool, optional
SHADE      = True
#Use more contour levels and a different color palette:
N_LEVELS   = 100
#If True, draw the cumulative distribution estimated by the kde.
#CUMULATIVE : bool, optional
CUMULATIVE = False

#test split ratio
train_val_split = 0.1
#bandwith parameter for the KDE Masks / Variance of the Gaussian Density

#hier die Werte 
experiment_info = {
        'memory'                    :   1,
        'search'                    :   1,
        #color
        'color'                     :   1,
        'grayscale'                 :   1,
        #experiment type
        'control'                   :   1,
        'peripheral_LP'             :   1,
        'peripheral_HP'             :   1,
        'foveal_LP'                 :   1,
        'foveal_HP'                 :   1,
        #search type
        'target_not_present'        :   1,
        'expectedlocation'          :   1,
        'randomlocation'            :   1
     }

def load_fixationDataMemory(dir_fixations):
  """Load the experiment datatable with pandas and
  filter anly valid rows and selecte few columns
  
  Arguments:
     dir_fixations: Datapath for the fixation DataFrame
     
  Returns:
     Filtered DataFrame
  """
  all_data = pd.read_table(dir_fixations) 
 
  filtered_data = all_data[(
      all_data["fixinvalid"] != 1)  &
  (all_data["sacinvalid"] != 1)  
  ].loc[:,["subject","colorimages","imageid","masktype","maskregion",
  "fixposx","fixposy","targetpresent", "expectedlocation", "fixno"]]
 
  return filtered_data


def filter_fixation_class(filtered_data, 
                          type_of_experiment,
                          type_of_search,
                          memory, 
                          color
                          ):
    """This function filter the fixation for experiment type and gives back a
    DataFrame with a single Experiment type
    
    Arguments:
        filtered_data: DataFrame
        
        type_of_experiment:  (String)
                            'control',  
                            'peripheral_LP',
                            'peripheral_HP'
                            'foveal_LP',
                            'foveal_HP' 
                            
        search_type: (String)  (String)
                               'target_not_present'     
                               'expectedlocation'        
                               'randomlocation'                              
                            
        memory: (String)        memory or search                    
                            
        color:              Bool
        
    Returns:
        Filtered DataFrame with a single Experiment type 
    """
    experiment = type_of_experiment

    
    if memory:
        experiment_class = filtered_data[ 
                (filtered_data["colorimages"] == color) &
                (filtered_data["masktype"] == experiment['masktype']) & 
                (filtered_data["maskregion"] == experiment['maskregion'])
                ].loc[:,
                ["subject","imageid","fixposx","fixposy",
               "colorimages", "fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"
               ]]
    elif memory == 0:
        search = type_of_search
        experiment_class =  filtered_data[ 
               (filtered_data["colorimages"] == color) &
               (filtered_data["masktype"] == experiment['masktype']) & 
               (filtered_data["maskregion"] == experiment['maskregion']) &
               (filtered_data["targetpresent"] == search["targetpresent"]) &
               (filtered_data["expectedlocation"] == search["expectedlocation"])
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]]
        
    return experiment_class

    
def combining_imageWith_fixations(filtered_experiment_class, 
                        dir_images,
                        memory,
                        search_type):
    """Create Tripplets  [Imagename (String),
    EyeMovements (DataFrame), Imagepath (path)]
    
    
    Argument:
        filtered_experiment_class: (DataFrame) Eye Movements
        
        dir_images: (Path)      Image Directory
        
        memory: (String)        memory or search
        
        search_type: (String)  (String)
                               'target_not_present'     
                               'expectedlocation'        
                               'randomlocation'  
        
    Returns:
        Tripplets (Lists)
    
    """
    path =0
    image_paths = get_img_paths(dir_images)
    
    list_of_masks = []
        
    for i, image in enumerate(np.unique(filtered_experiment_class["imageid"]) ):  
        for image_path in image_paths:
            if memory ==1:
                #in case of memory
                if int(re.findall(r'\d+',
                                  os.path.basename(image_path))[0]) ==  image:
                    path = image_path
            elif memory==0:
                
                if search_type == "target_not_present":
                    match = "L_"
                elif search_type == 'expectedlocation':
                    match = "E_"
                else:
                    match = "U_"
                                   
                #in case of search
                if  int(re.findall(r'\d+', 
           os.path.basename(image_path))[0]) == image and match in image_path:
                    path = image_path

        dataframe_temp =\
        filtered_experiment_class[filtered_experiment_class["imageid"] == image]
        
        if path==0:
            raise ValueError("wrong image path !")
        
        list_of_masks.append([str(i+1)+".png",
                             path,
                             dataframe_temp])
    return list_of_masks

def createMask(image_turple,
               type_e,
               train_list, 
               validate_list,
               output_path,
               height,
               width):
    """Create one Image and One Mask and save them  in the output + own directory
    The images are saved in train or validation directory. 
    
    Arguments:
        image:                    a  tripplet  [Imagename (String),
                                                  EyeMovements (DataFrame),
                                                  Imagepath (path)]
        train_list:              (list) list of ids for train data
        
        validate_list:           (list) list of ids for test data
        
        output_path:           (path) Output Pathe where the images are saved
        
        height:                   (int)  height of the image and mask
        
        width:                     (int)  wiidth of the image and mask
       
        
    Returns: 
        None
    """
    
    image_id = os.path.basename(image_turple[1])
    image_id_number = get_first_number_from_path(image_turple[1])
   
    if image_id_number in train_list:
      directory_name = "/training"
    elif image_id_number in validate_list:
      directory_name = "/validation"      
      
   
    if image_turple[2].iloc[:,2].min() > 0:
              
        mask = kde2D_plot(image_turple[2].iloc[:,2],
                          image_turple[2].iloc[:,3],
                          BANDWITH,
                          (height, width)
                          )
        
        single_channel_mask = mask[:,:,1]
        
        save_image(single_channel_mask,
                   output_path+directory_name+"/masks/ma/"+type_e+image_id)

               
        image = resize(imageio.imread(image_turple[1]),
                           (height, width), 
                           preserve_range = True)
        
        save_image(image, 
                   output_path+directory_name+"/images/im/"+type_e+image_id)
        
#  

#sns.jointplot(x="x", y="y", data=df, kind="kde")
#
#

          
def create_dir_of_data(images_withFixations, 
                       output_path,
                       height,
                       width,
                       train_val_split,
                       seed,
                       image_type,
                       experiment_type,
                       search_experiment_type
                       ):
    """create masks and images for given experiment.
    It will create training and validation directory in which the created
    images and masks are saved. 
    
    
    Arguments:
        images_withFixations: List of tripplets  [Imagename (String),
                                                  EyeMovements (DataFrame),
                                                  Imagepath (path)]
        
        output_path:           (path) Output Pathe where the images are saved
        
        height:                   (int)  height of the image and mask
        
        width:                     (int)  wiidth of the image and mask
      
        train_val_split:           (int) splitting validation ratio
        
        seed:                      (int) random seed
        
        image_type:                (string) memory or search
        
        experiment_type:          (String)
                                   'control',  
                                   'peripheral_LP',
                                   'peripheral_HP'
                                   'foveal_LP',
                                   'foveal_HP'   
                       
        search_experiment_type:     (String)
                                   'target_not_present'     
                                   'expectedlocation'        
                                   'randomlocation'    
        
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
    
    images_names_list = [get_first_number_from_path(images[1]) for images in images_withFixations]
    unque_images_length = len(np.unique(images_names_list))
    splitting_imageindex =  int(unque_images_length * (1- train_val_split)) 
      
    print("Number of created Images: ",unque_images_length)
    
    np.random.seed(seed)
    np.random.shuffle(images_names_list)
    
    train_list = images_names_list[:splitting_imageindex]
    validate_list = images_names_list[splitting_imageindex:]
        
    if "memory" in image_type:
        type_e = image_type +"_" +experiment_type + "_"
        output_path += "BW_"+str(BANDWITH)+"_memory_" + experiment_type  
        
    elif "search" in image_type:
        type_e = image_type +"_" \
        +experiment_type + "_" +search_experiment_type + "_"
        output_path +=  "BW_"+str(BANDWITH)+"_search_" + experiment_type + "_" +search_experiment_type
        
    print("Dir: ",output_path)
    
    create_dir(output_path+"/training/images/im")
    create_dir(output_path+"/training/masks/ma")
    create_dir(output_path+"/validation/images/im")
    create_dir(output_path+"/validation/masks/ma")
    
    """Parallelizing all images and masks in one experiment"""    
    Parallel(n_jobs=-1)(delayed(createMask)(
             image,
             type_e, 
             train_list,
             validate_list,
             output_path,
             height, 
             width) for image in images_withFixations)    
        
         
def create_masks_images_from_data(dir_fixations,
                                  dir_images,
                                  output_path,
                                  exp_type,
                                  search_type,
                                  train_val_split,
                                  seed,
                                  height,
                                  width,
                                  image_type,
                                  experiment_type,
                                  search_experiment_type
                                  ):
    """This is the main function to build masks from the fixation-data.
    
    It creates the output folder with the subdirectory training and validation
    data sets and also the image and mask subdirectory.
    
    Arguments:
       dir_fixations: (path) to the Datatable with the fixations
       
       dir_images:   (paths)  "original" path to the images from experiment 
       
       output_path:  (path)  the path name where the training images are saved
       
       exp_type:     (dictionary)
         
       search_type: (dictionary)
         
       train_val_split:(int) splitting validation ratio
        
       seed: (int) random seed
        
      image_type: (string)  'memory_color'  
                      'memory_grayscale'
                      'search_color'    
                      'search_grayscale' 
        
       experiment_type:       (String)
                       'control',  
                       'peripheral_LP',
                       'peripheral_HP'
                       'foveal_LP',
                       'foveal_HP'   
                       
       search_experiment_type:     (String)
                       'target_not_present'     
                        'expectedlocation'        
                        'randomlocation'                      
       
    """
    if "color" in dir_images:
        colortype = 1
    else:
        colortype = 0
        
    if "memory" in dir_images:
        memory = 1
    else:
        memory = 0

    filtered_data = load_fixationDataMemory(dir_fixations)
    filtered_experiment_class = filter_fixation_class(filtered_data,
                                                      exp_type, 
                                                      search_type,
                                                      memory,
                                                      colortype)
    images_withFixations = combining_imageWith_fixations(filtered_experiment_class, 
                                               dir_images,
                                               memory,
                                               search_type
                                               )
    create_dir_of_data(images_withFixations, 
                       output_path,
                       height,
                       width,
                       train_val_split,
                       seed,
                       image_type,
                       experiment_type,
                       search_experiment_type
                       )


def prepare_images_for_one_experiment_type(
                                image_type,
                                experiment_type,
                                search_experiment_type,
                                height,
                                width,
                                output_path,
                                train_val_split                                           
        ):
  
    """Prepares Inputs for different experiment type and
     it creates the output folder with the subdirectory 
     training and validation data sets and also the image and mask subdirectory.
    
    
    Arguments:
       dir_fixations: (path) to the Datatable with the fixations
       
       dir_images:   (paths)  "original" path to the images from experiment 
       
       output_path:  (path)  the path name where the training images are saved
       
       exp_type:     (dictionary)
         
       search_type: (dictionary)

       seed: (int) random seed
        
      image_type: (string)  'memory_color'  
                      'memory_grayscale'
                      'search_color'    
                      'search_grayscale' 
        
       experiment_type:       (String)
                       'control',  
                       'peripheral_LP',
                       'peripheral_HP'
                       'foveal_LP',
                       'foveal_HP'   
                       
       search_experiment_type:     (String)
                       'target_not_present'     
                        'expectedlocation'        
                        'randomlocation'   
       train_val_split:(int) splitting validation ratio                 
    """
    #image directory choose between color or grayscale here and only original 
    directorys_images = {
            'memory_color':     MEMORY_COLOR,
            'memory_grayscale': MEMORY_GRAYSCALE,
            'search_color':     SEARCH_COLOR,
            'search_grayscale': SEARCH_GRAYSCALE      
            }
    dir_images = directorys_images[image_type]
    
    blickdaten = {
            'memory' : MEMORY,
            'search' : SEARCH       
            }
    
    if 'memory' in dir_images:
        dir_fixations = blickdaten['memory']
    elif 'search' in dir_images:
        dir_fixations = blickdaten['search']
    #choose the experiment type
    experiment_types = {
        'control'                :   {'masktype' : 0, 'maskregion': 0 },
        'peripheral_LP'          :   {'masktype' : 1, 'maskregion': 1 },
        'peripheral_HP'          :   {'masktype' : 2, 'maskregion': 1 },
        'foveal_LP'              :   {'masktype' : 1, 'maskregion': 2 },
        'foveal_HP'              :   {'masktype' : 2, 'maskregion': 2 }
     }
    #experiment type
    exp_type= experiment_types[experiment_type]
    #if the experiment is search type choose search type
    search_types = {
    'target_not_present'      :   {'targetpresent' : 0, 'expectedlocation': 0 },
    'expectedlocation'        :   {'targetpresent' : 1, 'expectedlocation': 1 },
    'randomlocation'          :   {'targetpresent' : 1, 'expectedlocation': 0 },
    '0'                       :   "",
    }
    
#    output_path +=  "/" + experiment_type 
    
    #seach type
    if search_experiment_type:
        search_type = search_types[search_experiment_type]
    else:
        search_type = 0
    
    create_masks_images_from_data(dir_fixations,
                              dir_images,
                              output_path,
                              exp_type,
                              search_type,
                              train_val_split,
                              seed,
                              height,
                              width,
                              image_type,
                              experiment_type,
                              search_experiment_type
                              )


def get_keys(experiment_info, data):
    """Searching the keys in the experiment and returns all the keys needed for
    one experiment
    
    Arguments:
        experiment_info : a dictionary contains the experiments
        data            : (json) contains all the experiment type as dictionary
        
    Returns:
        Keys: (list) Experiments
    
    """
              
    list_of_expected_exp_keys = get_keyname_from_true_dict(experiment_info)
                
    exp_search_memory    = []
    color_or_grayscale   = []
    per_or_zen_pass      = []
    expectation          = []
    
    for key_name in list_of_expected_exp_keys:
        
        #collecting all the key containing the given exp list
        
        if "memory" in key_name or "search" in key_name:
            exp_search_memory    += get_keynames_contain_keyword(data, key_name)    
    
        if "color" in key_name or "grayscale" in key_name:
            color_or_grayscale   += get_keynames_contain_keyword(data, key_name)    
            
    
        if "control" in key_name or "peripheral_LP" in key_name or \
        "peripheral_HP" in key_name or "foveal_LP" in key_name\
        or "foveal_HP" in key_name:
             per_or_zen_pass     += get_keynames_contain_keyword(data, key_name)     
              
        if "target_not_present" in key_name or \
        "expectedlocation" in key_name or "randomlocation" in key_name:
             expectation     += get_keynames_contain_keyword(data, key_name)    


    intersection_off_color_and_type = intersection_of_2_list(
                                   exp_search_memory,
                                   color_or_grayscale
                                   )
    
    intersection_off_filter_and_intersection = intersection_of_2_list(
                                    per_or_zen_pass,
                                    intersection_off_color_and_type
                                    )
    
    searches = intersection_of_2_list(
                                    expectation,
                                    intersection_off_filter_and_intersection
                                    )
    
    memorys = get_word_contains_in_list(
                                    intersection_off_filter_and_intersection , 
                                    "memory")
        
    return sum([memorys,searches],[])
    

def processInput(data, key):
    """Use the given key and data dictionary to create to create 
    one experiment type. This function would be parallelized.
    
    Arguments:
        data      : contains all the experiment type as dictionary
        key       : experiment key  
        
    Returns:
        None
    
    """
    print("Creating: ",key, ":")

    prepare_images_for_one_experiment_type(
                            data[key]["image_type"],
                            data[key]["experiment_type"],
                            data[key]["search_type"],
                            height,
                            width,
                            output_path,
                            train_val_split
                            )
    
def main():    

    """All possible experiments are listed in a json file"""
    with open("config_file.json") as data_file:
        data = json.load(data_file)
           
    """here we get the keys fot the needed experiments"""
    keys = get_keys(experiment_info, data)    
    
    print("all found experiments: (",len(keys),") ", keys)
    
    """parallelizing the creating process if cpu is powerfull"""
    Parallel(n_jobs=1)(delayed(processInput)(data, key) for key in keys)
           

if __name__ == '__main__':
    main()
 


