#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:02:21 2020

@author: mendel
"""
import os
import matplotlib.pyplot as plt
import imageio


from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np

def get_model_paths(start_dir, extensions = ['png','jpeg','jpg','pneg','peng']):
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
  
  


 






  
img_path = '/mnt/data/DATA/dataset/UNET/prepared_dataset/set_salicon_bw_30/train/images/im'
mask_path = '/mnt/data/DATA/dataset/UNET/prepared_dataset/set_salicon_bw_30/train/masks/ma'
save_path = '/home/mendel/Videos/pytorch-CycleGAN-and-pix2pix/datasets/facades/train'



img_path_test = '/mnt/data/DATA/dataset/UNET/prepared_dataset/set_salicon_bw_30/test/images/im'
mask_path_test = '/mnt/data/DATA/dataset/UNET/prepared_dataset/set_salicon_bw_30/test/masks/ma'
save_path_test = '/home/mendel/Videos/pytorch-CycleGAN-and-pix2pix/datasets/facades/test'


img_path_val = '/mnt/data/DATA/dataset/UNET/prepared_dataset/ANKES_BW_30_memory_control/train/images/im'
mask_path_val = '/mnt/data/DATA/dataset/UNET/prepared_dataset/ANKES_BW_30_memory_control/train/masks/ma'
save_path_val = '/home/mendel/Videos/pytorch-CycleGAN-and-pix2pix/datasets/facades/val' 



def prepare_for_gan(img_path, mask_path, output_path):
  

  img_paths = get_model_paths(img_path)
  
  mask_paths = get_model_paths(mask_path)
  for i, path in enumerate(zip(img_paths,mask_paths)):
    
  
    img = imageio.imread(path[0]) 
    
    mask = imageio.imread(path[1])
    
    new_mask = np.zeros((mask.shape[0],mask.shape[1],(3)))
    
    new_mask[:,:,0] = mask[:,:,0]
    new_mask[:,:,1] = mask[:,:,0]
    new_mask[:,:,2] = mask[:,:,0]
    
    
    new_image = np.concatenate([new_mask,img], axis = 1)
    
    
    image_resized = resize(new_image, (new_image.shape[0] // 4, new_image.shape[1] // 4),
                         anti_aliasing=True)
  
    
    imageio.imwrite(output_path+'/'+str(i)+'.jpg', image_resized / 255)




prepare_for_gan(img_path,mask_path, save_path)
prepare_for_gan(img_path_test,mask_path_test, save_path_test)
prepare_for_gan(img_path_val,mask_path_val, save_path_val)

#
#img_paths = get_model_paths(img_path_test)
#
#mask_paths = get_model_paths(mask_path)
#
#for i, path in enumerate(zip(img_paths,mask_paths)):
#
#  img = imageio.imread(path[0])
#  mask = imageio.imread(path[1])
#  
#  
#  new_image = np.concatenate([mask,img], axis = 1)
#  
#  imageio.imwrite(save_path+'/'+str(i)+'.jpg', new_image)