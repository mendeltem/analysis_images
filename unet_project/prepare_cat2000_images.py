#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 19:40:03 2019

@author: MUT, DD

the benchmark data set containing 2000 images from 20 different categories 
with eye tracking data from 24 observers can be splitted randomly in to 
test and train folders. 

source : http://saliency.mit.edu/results_cat2000.html

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
    
    

#image and mask paths with the data set
image_path  =   config["cat2000_TRAIN_IMAGE"]
mask_path   =   config["cat2000_FIXATIONMAPS"]
#output file would be created
output_path = config["DATASET"] +"set_big_data_1"

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


def create_image_mask(image_dir,
                      mask_dir,
                      output_path,
                      height,
                      width,
                      train_list,
                      test_list):
    """creates dataset folders with resized images and split the dataset in
    to train and test dataset
    
    Arguments:
        image_path:             (path) path with images
        
        mask_path:              (path) path with masks
        
        output_path:           (path) Output Pathe where the train and test 
                                      folder with the images/masks are  saved
        
        height:                   (int)  height of the image and mask
        
        width:                     (int)  wiidth of the image and mask
           
        train_list:               (list) contains the names of the train images
        
        test_list:                (list) contains the names of the test images
        
    Returns: 
        None
    """

    if os.path.dirname(image_dir).split('/')[-1] == \
    os.path.dirname(mask_dir).split('/')[-1]:
        
        #loading the images
        image = imageio.imread(image_dir)
        mask = imageio.imread(mask_dir)
        
        #resicing and casting tha types
        resized_image = resize(image, (height, width))
        resized_image = resized_image.astype(np.float64) / np.max(resized_image)
        resized_image = (resized_image * 255).astype(np.uint8)
        
        resized_mask  = resize(mask, (height, width))
        resized_mask = resized_mask.astype(np.float64) / np.max(resized_mask)
        resized_mask = (resized_mask * 255).astype(np.uint8)
               
        #new file names needed
        basename = os.path.basename(image_dir)
        filename = os.path.dirname(image_dir).split('/')[-1]
        
        #check if the file name is in test or train list
        if filename+"_"+basename in train_list:
            directory_name = "training"
        elif filename+"_"+basename in test_list:     
            directory_name = "test"  
            
        #creating the filder and saving the images
        imageio.imwrite(
                output_path+"/"+directory_name+"/images/"+filename+"_"+basename, 
                resized_image)
        
        imageio.imwrite(
                output_path+"/"+directory_name+"/masks/"+filename+"_"+basename, 
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
      
    name_list = []
    
    for img_dir in image_dirs:
        basename = os.path.basename(img_dir)
        filename = os.path.dirname(img_dir).split('/')[-1]
        name_list.append(filename+"_"+basename)
    
    
    number_of_images = len(image_dirs)
    number_of_masks = len(get_img_paths(mask_path))
    
    
    if number_of_images != number_of_masks:
        raise Exception("number of images and masks should be same")
    
    
    splitting_imageindex =  int(number_of_images * (1- train_test_split)) 
    
    np.random.seed(seed)
    np.random.shuffle(name_list)
        
    train_list = name_list[:splitting_imageindex]
    test_list = name_list[splitting_imageindex:]
        
    
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
                        image_dir,
                        mask_dir,
                        output_path,
                        height,
                        width,
                        train_list,
                        test_list
                        ) for image_dir, mask_dir in zip(image_dirs,mask_dirs))    
    
    
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
 