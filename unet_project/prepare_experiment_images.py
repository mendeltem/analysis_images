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
warnings.filterwarnings("ignore", category = UserWarning)
warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)

#from modules.library import get_img_paths

#Fixed Value for the Plot
#monitor size
my_dpi = 90
#path length for ploting kde images
pad_length = 27
#radnom seed
seed = 0
random.seed(seed)
np.random.seed(seed=seed)
"""Parameter for the process"""
#widh and height
height = 768
width = 1024
#output dir
output_path = "dataset/set_06"

#experiment data tables
MEMORY ='blickdaten/finalresulttab_funcsac_SFC_memory.dat'
SEARCH = 'blickdaten/finalresulttab_funcsac_SFC_search.dat'   

#input images paths for different imagetypes
MEMORY_COLOR = 'images/color/session1_memory/original'
MEMORY_GRAYSCALE = 'images/grayscale/session1_memory/original'
SEARCH_COLOR= 'images/color/session2_search/original'
SEARCH_GRAYSCALE= 'images/grayscale/session2_search/original' 

#seaborn.pydata.org/generated/seaborn.kdeplot.html


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
exeperiment_info = {
        'memory'                    :   1,
        'search'                    :   1,
        #color
        'color'                     :   1,
        'grayscale'                 :   1,
        #experiment type
        'control'                   :   1,
        'peripher_lowpassfilter'    :   1,
        'peripher_highpassfilter'   :   1,
        'zentral_lowpassfilter'     :   1,
        'zentral_highpassfilter'    :   1,
        #search type
        'target_not_present'        :   1,
        'expectedlocation'          :   1,
        'randomlocation'            :   1
     }

def get_img_paths(start_dir, extensions = ['png','jpeg','jpg','pneg']):
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
                            'peripher_lowpassfilter',
                            'peripher_highpassfilter'
                            'zentral_lowpassfilter',
                            'zentral_highpassfilter' 
                            
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
        exeperiment_class = filtered_data[ 
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
        exeperiment_class =  filtered_data[ 
               (filtered_data["colorimages"] == color) &
               (filtered_data["masktype"] == experiment['masktype']) & 
               (filtered_data["maskregion"] == experiment['maskregion']) &
               (filtered_data["targetpresent"] == search["targetpresent"]) &
               (filtered_data["expectedlocation"] == search["expectedlocation"])
               ].loc[:,["subject","imageid","fixposx","fixposy",
               "colorimages","fixno",
               "masktype","maskregion",
               "targetpresent", "expectedlocation"]]
        
    return exeperiment_class

    
def imageWith_fixations(filtered_experiment_class, 
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

def createMask(image,
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
    image_id = os.path.basename(image[1])
    image_id_number = int(re.search(r'\d+', 
                                    os.path.basename(image[1])).group())  
    
    if image_id_number in train_list:
      directory_name = "training"
      #print("Image Training ",os.path.basename(image[1]))
    elif image_id_number in validate_list:
      directory_name = "validation"      
      #print("Image Validation ",os.path.basename(image[1]))
      
    #print(image[2].iloc[:,2].min())  
    if image[2].iloc[:,2].min() > 0:
      
        plt.figure(figsize=(width/my_dpi, height/my_dpi), dpi=my_dpi)
        fig = sns.kdeplot(image[2].iloc[:,2],
                    image[2].iloc[:,3], 
                    kernel = KERNEL,
                    bw= BANDWITH,
                    cmap = plt.cm.gray,
                    shade=SHADE, 
                    n_levels = N_LEVELS,
                    legend= False,
                    cumulative= CUMULATIVE
                    )
        
        plt.close()
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
        image1 = np.fromstring(
                figure.canvas.tostring_rgb(),dtype=np.uint8,sep='')
        image1 = image1.reshape(
                figure.canvas.get_width_height()[::-1] + (3,))
        imageio.imwrite(
                output_path+"/"+directory_name+"/masks/"+type_e+image_id, 
                image1[:,:,0])          
        print(output_path+"/"+directory_name+"/masks/"+type_e+image_id)
        image_original = imageio.imread(image[1])

        res_image = resize(image_original,
                           (height, width), 
                           preserve_range = True)
        
        
        res_image = res_image.astype(np.uint8)

        imageio.imwrite(
                output_path+"/"+directory_name+"/images/"+type_e+image_id, 
                res_image)
        print(output_path+"/"+directory_name+"/images/"+type_e+image_id)
            
def create_image_masks(images_withFixations, 
                       output_path,
                       height,
                       width,
                       train_val_split,
                       seed,
                       image_type,
                       exeperiment_type,
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
        
        exeperiment_type:          (String)
                                   'control',  
                                   'peripher_lowpassfilter',
                                   'peripher_highpassfilter'
                                   'zentral_lowpassfilter',
                                   'zentral_highpassfilter'   
                       
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
    
    images_names_list = [int(re.search(r'\d+', 
      os.path.basename(images[1])).group()) for images in images_withFixations]
    unque_images_length = len(np.unique(images_names_list))
    splitting_imageindex =  int(unque_images_length * (1- train_val_split)) 
    
    
    print("Number of created Images: ",unque_images_length)
    
    np.random.seed(seed)
    np.random.shuffle(images_names_list)
    
    train_list = images_names_list[:splitting_imageindex]
    validate_list = images_names_list[splitting_imageindex:]
        
    #os.chdir(output_path)    
    if not os.path.exists(output_path+"/training/images"):
        os.makedirs(output_path+"/training/images")
    
    if not os.path.exists(output_path+"/training/masks"):
        os.makedirs(output_path+"/training/masks")
    
    #os.chdir(output_path)
    if not os.path.exists(output_path+"/validation/images"):
        os.makedirs(output_path+"/validation/images")
    
    if not os.path.exists(output_path+"/validation/masks"):
        os.makedirs(output_path+"/validation/masks")
    
    if "memory" in image_type:
        type_e = image_type +"_" +exeperiment_type + "_"
    elif "search" in image_type:
        type_e = image_type +"_" \
        +exeperiment_type + "_" +search_experiment_type + "_"
        
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
                                  exeperiment_type,
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
        
       exeperiment_type:       (String)
                       'control',  
                       'peripher_lowpassfilter',
                       'peripher_highpassfilter'
                       'zentral_lowpassfilter',
                       'zentral_highpassfilter'   
                       
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
    images_withFixations = imageWith_fixations(filtered_experiment_class, 
                                               dir_images,
                                               memory,
                                               search_type
                                               )
    create_image_masks(images_withFixations, 
                       output_path,
                       height,
                       width,
                       train_val_split,
                       seed,
                       image_type,
                       exeperiment_type,
                       search_experiment_type
                       )

def prepare_images_for_one_experiment_type(
                                image_type,
                                exeperiment_type,
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
        
       exeperiment_type:       (String)
                       'control',  
                       'peripher_lowpassfilter',
                       'peripher_highpassfilter'
                       'zentral_lowpassfilter',
                       'zentral_highpassfilter'   
                       
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
    exeperiment_types = {
        'control'                :   {'masktype' : 0, 'maskregion': 0 },
        'peripher_lowpassfilter'  :   {'masktype' : 1, 'maskregion': 1 },
        'peripher_highpassfilter' :   {'masktype' : 2, 'maskregion': 1 },
        'zentral_lowpassfilter'   :   {'masktype' : 1, 'maskregion': 2 },
        'zentral_highpassfilter'  :   {'masktype' : 2, 'maskregion': 2 }
     }
    #experiment type
    exp_type= exeperiment_types[exeperiment_type]
    #if the experiment is search type choose search type
    search_types = {
    'target_not_present'      :   {'targetpresent' : 0, 'expectedlocation': 0 },
    'expectedlocation'        :   {'targetpresent' : 1, 'expectedlocation': 1 },
    'randomlocation'          :   {'targetpresent' : 1, 'expectedlocation': 0 },
    '0'                       :   "",
    }
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
                              exeperiment_type,
                              search_experiment_type
                              )


def get_keys(exeperiment_info, data):
    """Searching the keys in the experiment and returns all the keys needed for
    one experiment
    
    Arguments:
        experiment_info : a dictionary contains the experiments
        data            : (json) contains all the experiment type as dictionary
        
    Returns:
        Keys: (list) Experiments
    
    """
    list_of_expected = []
    
    for i, info in enumerate(list(exeperiment_info.keys())):
        if exeperiment_info[info]:
            list_of_expected.append(info)
  
    exp_search_memory =[]
    color_or_grayscale = []
    per_or_zen_pass  = []
    expectation      = []
    
    for i,l in enumerate(list_of_expected):
        if "search" in l or "memory" in l:
            exp_search_memory += (([key for i, 
                          key in enumerate( list(data.keys())) if l in key]))
        if "color" in l or "grayscale" in l:
            color_or_grayscale += (([key for i, 
                            key in enumerate( list(data.keys())) if l in key]))
        if "control" in l or "peripher_lowpassfilter" in l or \
        "peripher_highpassfilter" in l or "zentral_lowpassfilter" in l \
        or "zentral_highpassfilter" in l:
            per_or_zen_pass += (([key for i, 
                          key in enumerate( list(data.keys())) if l in key]))
          
        if "target_not_present" in l or \
        "expectedlocation" in l or "randomlocation" in l:
            expectation += (([key for i, 
                          key in enumerate( list(data.keys())) if l in key]))


    intersection_off_color_and_type = \
    [value for value in exp_search_memory\
     if value in color_or_grayscale] 
    
    intersection_off_filter_and_intersection = \
    [value for value in per_or_zen_pass \
     if value in intersection_off_color_and_type] 
    
    memorys = [i for i in \
               intersection_off_filter_and_intersection if "memory" in i]
    
    searches = \
    [value for value in expectation \
     if value in intersection_off_filter_and_intersection] 
    
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
    with open("experiment_list.json") as data_file:
        data = json.load(data_file)
           
    """here we get the keys fot the needed experiments"""
    keys = get_keys(exeperiment_info, data)    
    
    print("all found experiments: (",len(keys),") ", keys)
    
    """parallelizing the creating process if cpu is powerfull"""
    Parallel(n_jobs=1)(delayed(processInput)(data, key) for key in keys)
           

if __name__ == '__main__':
    main()
 


    

    