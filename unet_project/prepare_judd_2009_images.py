#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 19:40:03 2019

@author: MUT, DD

collected eye tracking data of 15 viewers on 1003 images

source to download: http://people.csail.mit.edu/tjudd/WherePeopleLook/index.html

"""
import os
import imageio
import numpy as np
import json
from joblib import Parallel, delayed
from skimage.transform import resize
import warnings
warnings.filterwarnings("ignore", category = UserWarning)
warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)


with open("input_file.json") as data_file:
    config = json.load(data_file)
    

#experiment data tables
image_path                  = config["Judd_2009_TRAIN_IMAGES"]
mask_path                   = config["Judd_2009_FIXATIONMAPS"]
    
#output file would be created
output_path                 =  config["DATASET"] + "/set_judd_2009_1"

#output image size
height = 600
width  = 800 
#test to train data ratio
train_test_split = 0.1
#seed for random 
seed  = 0

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


def name(img_dir, mask_dirs):
  """Binding the same masks and images together with tuple
  a tuple with image and mask directorys
  
    Arguments:
        img_dir:      PATH to Image
        mask_dirs:    PATHs to Masks

    Returns:     
      img_dir:       PATHS
      mask_dir:      PATHS
      img_name:      string basename
      
  
  """
  img_name = os.path.basename(img_dir).rsplit('.jpeg', 1)[0]  
  mask_dir = [mask_dir for mask_dir in mask_dirs \
    if os.path.basename(mask_dir).rsplit('_fixMap.jpg', 1)[0]  == img_name][0]
  
  return img_dir, mask_dir, img_name


def create_image_mask(output_path,
                        height,
                        width,
                        data_turple,
                        train_test
                        ):
    """creates dataset folders with resized images and split the dataset in
    to train and test dataset
    
    Arguments:
        
        output_path:           (path) Output Pathe where the train and test 
                                      folder with the images/masks are  saved
        
        height:                   (int)  height of the image and mask
        
        width:                     (int)  wiidth of the image and mask
           
        data_turple:               (turple) (image paths,
                                             maske paths,
                                             base names
                                          )
        
        
        train_test:                (bool)  True -> Train, False -> Test
                                    
        
    Returns: 
        None
    """
    
    img_dir = data_turple[0]
    mask_dir = data_turple[1]
    base_name = data_turple[2]
    
    image = imageio.imread(img_dir)
    mask = imageio.imread(mask_dir)
    
    #resicing and casting tha types
    resized_image = resize(image, (height, width))
    resized_image = resized_image.astype(np.float64) / np.max(resized_image)
    resized_image = (resized_image * 255).astype(np.uint8)
        
    resized_mask  = resize(mask, (height, width))
    resized_mask = resized_mask.astype(np.float64) / np.max(resized_mask)
    resized_mask = (resized_mask * 255).astype(np.uint8)
               
    
    train_test        #check if the file name is in test or train list
    if train_test   == True:
        directory_name = "training"
    elif train_test == False:     
        directory_name = "test"    

    #creating the filder and saving the images
    imageio.imwrite(
            output_path+"/"+directory_name+"/images/"+base_name+".jpg", 
            resized_image)
    
    imageio.imwrite(
            output_path+"/"+directory_name+"/masks/"+base_name+".jpg", 
            resized_mask)
        
        
def prepare_images(image_path,
                   mask_path,
                   output_path,
                   height,
                   width,
                   train_test_split, 
                   seed=0):
    """creates dataset folders with resized images and split the dataset in
    to train and test dataset
    
    randomly split the names in to train and test lists.
    
    Arguments:
        image_path:             (path) path with images
        
        mask_path:              (path) path with masks
        
        output_path:           (path) Output Pathe where the train and test 
                                      folder with the images/masks are  saved
        
        height:                   (int)  height of the image and mask
        
        width:                     (int)  wiidth of the image and mask
        
        
        train_test_split:           (float) spliting ratio test to train
        
        
        seed:                      (int) random seed
       
        
    Returns: 
        None
    """
    
    image_dirs = [img_path for img_path in get_img_paths(image_path)\
                  if "Output" not in img_path]
    mask_dirs  = get_img_paths(mask_path)
    
    if len(image_dirs) == 0:
      raise ValueError("Wrong image Path !")

    if len(mask_dirs) == 0:
      raise ValueError("Wrong mask Path !")
            
    turple_dir = Parallel(n_jobs=-1)(delayed(name)(img_dir,mask_dirs)\
                    for img_dir in image_dirs)
    
    number_of_images = len(turple_dir)

    
    splitting_imageindex =  int(number_of_images * (1- train_test_split)) 
    
    
    #splitting in test and train sets randomly
    np.random.seed(seed)
    np.random.shuffle(turple_dir)
        
    train_list = turple_dir[:splitting_imageindex]
    test_list = turple_dir[splitting_imageindex:]
        
    
    #os.chdir(output_path)    
    if not os.path.exists(output_path+"/training/images"):
        os.makedirs(output_path+"/training/images")
    
    if not os.path.exists(output_path+"/training/masks"):
        os.makedirs(output_path+"/training/masks")
    
    #os.chdir(output_path)
    if not os.path.exists(output_path+"/test/images"):
        os.makedirs(output_path+"/test/images")
    
    if not os.path.exists(output_path+"/test/masks"):
        os.makedirs(output_path+"/test/masks")
      
    """Parallelizing all images and masks in one experiment"""    
    print("Creating Images and masks in", output_path)
    Parallel(n_jobs=-1)(delayed(create_image_mask)(
                        output_path,
                        height,
                        width,
                        train_turple,
                        train_test = True
                        ) for train_turple in train_list)  
    
    
    Parallel(n_jobs=-1)(delayed(create_image_mask)(
                    output_path,
                    height,
                    width,
                    test_turple,
                    train_test = False
                    ) for test_turple in test_list)   
    
    
def main():    

    """Images/Masks are resized and split in to train and test dataset"""
    prepare_images(image_path,
                   mask_path,
                   output_path,
                   height,
                   width,
                   train_test_split, 
                   seed)
           

if __name__ == '__main__':
    main()
 
