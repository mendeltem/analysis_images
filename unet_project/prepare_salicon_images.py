#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:31:26 2019

@author: MUT,DD

Create the training and validation datasets with kernel density estimation 

Dataset is from SALICON dataset

source: http://salicon.net/download/

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import imageio
from skimage.transform import resize
import json
#from joblib import Parallel, delayed
import warnings
import random 
from modules.library import  get_word_contains_in_list,kde2D_plot
from modules.library import  get_keynames_contain_keyword
from modules.library import  get_first_number_from_path,create_kde,save_image

#warnings.filterwarnings("ignore", category = UserWarning)
#warnings.filterwarnings("ignore", category = DeprecationWarning)
#warnings.filterwarnings("ignore", category = FutureWarning)
#Fixed Value for the Plot
#monitor size
my_dpi = 90
#path length for ploting kde images
pad_length = 27

#from modules.library import get_img_paths
with open("input_file.json") as data_file:
    config = json.load(data_file)
    
    
mask_path                     = config["SALICON_FIXATION"]
image_path                    = config["SALICON_TRAIN_IMAGES"]
output_path                   = config["DATASET"] + "set_salicon_bw_"


##image and mask paths with the data set
#image_path = "../images/SALICON/SALICON_stimu/train"
#mask_path = "../images/SALICON/fixations_train2014.json"
##output file would be created
#output_path = "../dataset/set_salicon_bw_test"

#output image size
#HEIGHT = 600
#WIDTH  = 800 
#test to train data ratio
train_test_split = 0.1
#seed for random 
SEED  = 1

SAMPLE_DISTANCE = 0

#Code for shape of kernel to fit with. Bivariate KDE can only use gaussian kernel.
#KERNEL : {‘gau’ | ‘cos’ | ‘biw’ | ‘epa’ | ‘tri’ | ‘triw’ }, optional
KERNEL     = "gau"
#Name of reference method to determine kernel size,
#scalar factor, or scalar for each dimension of the bivariate plot.
#Note that the underlying computational libraries have different 
#interperetations for this parameter: statsmodels uses it directly,
#but scipy treats it as a scaling factor for the standard deviation of the data.
#BANDWITH : {‘scott’ | ‘silverman’ | scalar | pair of scalars }, optional
#BANDWITH   = 30

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
plt.ioff()


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
  
  
def create_mask_image(
                      image_turples,
                      train_or_test_index, 
                      output_path,
                      bw
                      ):  
  """Create one mask and prepare one image
  
    Arguments:
        turple:   (name of the image, 
                   all fixations from the image,
                   heigh of the image, 
                   width of the image
                   )
        
        
        output_path:           (path) Output Pathe where the train and test 
                              folder with the images/masks are  saved

    Returns:
        a sorted list of all image paths starting from the root of the file
        system.
        
  """
  name              =  image_turples[0]
  fixations         =  image_turples[1]
  height            =  image_turples[2]
  width             =  image_turples[3]  
  image_dir         =  image_turples[4]  
  
  #initial
  mean_distance_x = 1
  var_distance_x = 1
  mean_distance_y = 1
  var_distance_y = 1
  SAMPLE_RATIO = 0.01
  
  if SAMPLE_DISTANCE  :

    #sampline from the fixations check if the var and mean distance is small
    while abs(mean_distance_x) > SAMPLE_DISTANCE or abs(var_distance_x) > \
    SAMPLE_DISTANCE or abs(mean_distance_y) > SAMPLE_DISTANCE or \
    abs(var_distance_y) > SAMPLE_DISTANCE:
      
      sample_size =  int(len(fixations) * SAMPLE_RATIO)
      sample_fixations = random.choices(fixations, k=sample_size)    
           
      x = [fixation[0] for fixation in sample_fixations]
      y = [fixation[1] for fixation in sample_fixations]
      mean_distance_x = 1 - abs(np.mean(x) /np.mean(x))
      mean_distance_y = 1 - abs(np.mean(y) /np.mean(y))
      var_distance_x  = 1 - abs(np.var(x)/np.var(x))
      var_distance_y  = 1 - abs(np.var(y)/np.var(y))
      SAMPLE_RATIO += 0.01
      
  else:
    x = [fixation[0] for fixation in fixations]
    y = [fixation[1] for fixation in fixations]
    
  print("sample_of_y",len(y))
  start = time.time()
  
  
  mask = kde2D_plot(y,
                    x,
                    bw,
                    (height, width)
                    )
  
   
  stop = time.time()
  print('Elapsed time for KDE PLOT: {:.2f} s'.format(stop - start))

  
  mask = mask.astype(np.float64) / np.max(mask)
  mask = (mask * 255).astype(np.uint8)
  
  if train_or_test_index   ==  True:
    directory_name = "train"
  elif train_or_test_index == False:
    directory_name = "test" 

  image = resize(imageio.imread(image_dir), (height,width))
  image = image.astype(np.float64) / np.max(image)
  image = (image * 255).astype(np.uint8)
  
  #creating the filder and saving the images
  imageio.imwrite(
          output_path+"/"+directory_name+"/images/im/"+name, 
          image)
  
  imageio.imwrite(
          output_path+"/"+directory_name+"/masks/ma/"+name, 
          mask)


def combine_image_with_fixations(data_image,data_annotation, image_path):
  """combine image id with fixations and save it in turple
  
  
    Arguments:
        data_image: directory with the image information

        data_annotation: directory with the fixation information


    Returns:
        turple:   (name of the image, 
                   all fixations from the image,
                   heigh of the image, 
                   width of the image,
                   image path
                   )
  """
  
  image_id  = data_image['id'] 
  name  = data_image['file_name'] 
  height  = data_image['height'] 
  width  = data_image['width'] 
  
  image_dir = [(i_dir) \
  for i_dir in get_img_paths(image_path) if os.path.basename(i_dir) ==name ][0]
  
  fixations = []

  for an in data_annotation:
    if an['image_id'] ==image_id:
      for fixation in an['fixations']:
        if fixation:
          fixations.append(fixation)
   
  print("image_id ", image_id, " has ", len(fixations), "fixations")      
  return (name,fixations,height, width, image_dir)  
  

def prepare_dataset(data,image_dir ,train_test_split,output_path, bw): 
  """Prepare to create the Dataset. Creates train test folder.
  Split the Images in to train test
  
  
    Arguments:
        data:      loaded ground truth data information

        image_dir:  (PATH)  path to the images


    Returns:
        None
  """   
  #os.chdir(output_path)    
  output_path += bw


  if not os.path.exists(output_path+"/train/images/im"):
      os.makedirs(output_path+"/train/images/im")
  
  if not os.path.exists(output_path+"/train/masks/ma"):
      os.makedirs(output_path+"/train/masks/ma")
  
  #os.chdir(output_path)
  if not os.path.exists(output_path+"/test/images/im"):
      os.makedirs(output_path+"/test/images/im")
  
  if not os.path.exists(output_path+"/test/masks/ma"):
      os.makedirs(output_path+"/test/masks/ma")  
  
  sns.set_style("darkgrid", {"axes.facecolor": ".9",
                         'figure.facecolor': 'white',
                          'axes.spines.bottom': False,
                           'axes.spines.left': False,
                           'axes.spines.right': False,
                           'axes.spines.top': False,
                         }) 

  data_images  =  data['images']
  data_annotation = data['annotations'] 
  number_of_images = len(data['images'])  
  
  data = 0
  
  splitting_imageindex =  int(number_of_images * (1- train_test_split)) 
  index = list(range(number_of_images))
  
  #splitting in test and train sets randomly
  np.random.seed(SEED)
  np.random.shuffle(index)
  train_list = index[:splitting_imageindex]
  test_list  = index[splitting_imageindex:]
  
  
  print("Number of train Image ", len(train_list))
  print("Number of test Image ", len(test_list))
  
  for n,data_image in enumerate(data_images):
      
    image_turples= combine_image_with_fixations(data_image,
                  data_annotation,
                  image_path
                  )
    
    if len(image_turples) == 0:
      raise ValueError("No image in Directory. Wrong image Path?")
    
    start = time.time()
    if n in train_list:
      
      train_or_test_index = True
      create_mask_image(
                    image_turples,
                    train_or_test_index,
                    output_path,
                    bw
                    )
    
    if n in test_list:
      train_or_test_index = False
      create_mask_image(
                    image_turples,
                    train_or_test_index,
                    output_path,
                    bw
                    )      
      stop = time.time()
      
      print('Elapsed time for the entire processing: {:.2f} s'\
            .format(stop - start))
    
def main():    

    """Load the ground truth from json file"""
    with open(mask_path) as data_file:
      data = json.load(data_file)
    
    print(data['info'])  
    
    bw =5
    
    
    for bw in [10,15,20,25,30,35,40]:
      print("creating bw:",bw)
      prepare_dataset(data,image_path,train_test_split, output_path,bw)


if __name__ == '__main__':
    main()
 