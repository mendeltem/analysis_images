#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:31:26 2019

@author: MUT,DD

Create the Training and test Datasets with Kernel Density Estimation
Masks from the eye movement experiment fixations.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import random

import seaborn as sns
import re
import imageio

from modules.library import get_img_paths
from modules.m_preprocess import copyfile

#Fixed Value for the Plot
#monitor size
my_dpi = 90
#path length for ploting kde images
pad_length = 27
#widh and height
w = 1024
h = 768
#radnom seed
seed = 42
random.seed(seed)
np.random.seed(seed=seed)
"""Parameter for the process"""
#output dir
output_path = "dataset/fixations"
#ratio of test set on total set
test_ratio = 0.1
#bandwith parameter for the KDE Masks / Variance of the Gaussian Density
bw = 30
#fixation data diractory
blickdaten = {
        'memory' : 'blickdaten/finalresulttab_funcsac_SFC_memory.dat',
        'search' : 'blickdaten/finalresulttab_funcsac_SFC_search.dat'
        }

dir_fixations = blickdaten['memory']
#image directory choose between color or grayscale here and only original
directorys_images = {
        'memory_color': 'images/color/session1_memory/original',
        'memory_grayscale': 'images/grayscale/session1_memory/original',
        'search_color': 'images/color/session2_search/original',
        'search_grayscale': 'images/grayscale/session2_search/original'
        }
dir_images = directorys_images['memory_color']
#choose the experiment type
exeperiment_type = {
    'control'                :   {'masktype' : 0, 'maskregion': 0 },
    'peripher_lowpassfilter'  :   {'masktype' : 1, 'maskregion': 1 },
    'peripher_highpassfilter' :   {'masktype' : 2, 'maskregion': 1 },
    'zentral_lowpassfilter'   :   {'masktype' : 1, 'maskregion': 2 },
    'zentral_highpassfilter'  :   {'masktype' : 2, 'maskregion': 2 }
 }
#experiment type
exp_type= exeperiment_type['control']
#if the experiment is search type choose search type
search_types = {
'target_not_present'      :   {'targetpresent' : 0, 'expectedlocation': 0 },
'expectedlocation'        :   {'targetpresent' : 1, 'expectedlocation': 1 },
'unexpectedlocation'      :   {'targetpresent' : 1, 'expectedlocation': 0 },
}
#seach type
search_type = search_types['expectedlocation']


def load_fixationDataMemory(dir_fixations):
  """Load the experiment datatable with pandas and
  filter anly valid rows and selecte few columns

  Argument:
     Datapath for the fixation DataFrame
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


def filter_fixation_class(filtered_data, type_of_experiment,type_of_search,memory, color):
    """This function filter the fixation for experiment type

    Argument:
        filtered_data:      DataFrame
        type_of_experiment:  (String)
                            'control',
                            'peripher_lowpassfilter',
                            'peripher_highpassfilter'
                            'zentral_lowpassfilter',
                            'zentral_highpassfilter'
        color:              Bool

    Returns:
        Filtered DataFrame with Experiment type
    """

    experiment = type_of_experiment
    search = type_of_search

    if memory:
        exeperiment_class = filtered_data[ (filtered_data["colorimages"] == color) &
                       (filtered_data["masktype"] == experiment['masktype']) &
                       (filtered_data["maskregion"] == experiment['maskregion'])
                       ].loc[:,["subject","imageid","fixposx","fixposy",
                       "colorimages", "fixno",
                       "masktype","maskregion",
                       "targetpresent", "expectedlocation"
                       ]]
    elif memory == 0:
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


def imageWith_fixations(filtered_experiment_class, dir_images, memory, search_type):
    """This function combinates the experiment DataFrame with Image

    Argument:
        DataFrame: filtered Experiment class
        Directory: Image Directory
    Returns:
        List of connected imagepaths with fixation DataFrames

    """
    image_paths = get_img_paths(dir_images)

    list_of_masks = []

    for i, image in enumerate(np.unique(filtered_experiment_class["imageid"]) ):
        for image_path in image_paths:
            if memory ==1:
                #in case of memory
                if int(re.findall(r'\d+', os.path.basename(image_path))[0]) ==  image:
                    path = image_path
            elif memory==0:

                if search_type == "target_not_present":
                    match = "L_"
                elif search_type == 'expectedlocation':
                    match = "E_"
                else:
                    match = "U_"

                #in case of search
                if  int(re.findall(r'\d+', os.path.basename(image_path))[0]) == image and match in image_path:
                    path = image_path

        dataframe_temp = filtered_experiment_class[filtered_experiment_class["imageid"] == image]

        list_of_masks.append([str(i+1)+".png",
                             path,
                             dataframe_temp])
    return list_of_masks


def create_image_masks(images_withFixations, output_path,bw=30,test_ratio = 0.5,seed=42):
    """create masks and images and splitt it in train and test set

    Arguments:
        images_withFixations: List of images with each own DataFrame and Path
        output_path: Output Pathe where the images are saved
        bw:   brandwith for KDE
        test_ratio: splitting ratio

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
    images_names_list = [int(re.search(r'\d+', os.path.basename(images[1])).group())
                          for images in images_withFixations]
    unque_images_length = len(np.unique(images_names_list))
    splitting_imageindex = int(unque_images_length * (1- test_ratio))

    np.random.seed(seed)
    np.random.shuffle(images_names_list)

    train_list = images_names_list[:splitting_imageindex]
    validate_list = images_names_list[splitting_imageindex:]

    #os.chdir(output_path)
    if os.path.exists(output_path+"/training/images"):
        shutil.rmtree(output_path+"/training/images")
    os.makedirs(output_path+"/training/images")

    if os.path.exists(output_path+"/training/masks"):
        shutil.rmtree(output_path+"/training/masks")
    os.makedirs(output_path+"/training/masks")

    #os.chdir(output_path)
    if os.path.exists(output_path+"/test/images"):
        shutil.rmtree(output_path+"/test/images")
    os.makedirs(output_path+"/test/images")

    if os.path.exists(output_path+"/test/masks"):
        shutil.rmtree(output_path+"/test/masks")
    os.makedirs(output_path+"/test/masks")

    #if np.sum(images_withFixations[-1][2].iloc[:,2]):

    for i, image in enumerate(images_withFixations):
        image_id = os.path.basename(image[1])
        image_id_number = int(re.search(r'\d+',
                              os.path.basename(image[1])).group())

        if image_id_number in train_list:
          directory_name = "training"
          print("Image Training ",os.path.basename(image[1]))
        elif image_id_number in validate_list:
          directory_name = "test"
          print("Image test ",os.path.basename(image[1]))

        print(image[2].iloc[:,2].min())
        if image[2].iloc[:,2].min() > 0:

            plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
            fig = sns.kdeplot(image[2].iloc[:,2],
                        image[2].iloc[:,3],
                        kernel = "gau",
                        bw= bw,
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
            image1 = np.fromstring(figure.canvas.tostring_rgb(),
                                   dtype=np.uint8,sep='')
            image1 = image1.reshape(figure.canvas.get_width_height()[::-1] + (3,))
            imageio.imwrite(output_path+"/"+directory_name+"/masks/"+image_id,
                            image1[:,:,0])

            copyfile(image[1],
                     output_path+"/"+directory_name+"/images/"+image_id)


def create_masks_images_from_data(dir_fixations,
                                  dir_images,
                                  output_path,
                                  exp_type,
                                  search_type,
                                  bw=30,
                                  test_ratio = 0.5,
                                  seed=42
                                  ):
    """This is the main function to build masks from the fixation-data.

    It creates the output folder with the subdirectory training and test
    data sets and also the imag and mask subdirectory.

    Arguments:
       dir_fixations:  path to the Datatable with the fixations
       dir_images:     "original" path to the images from experiment
       output_path:    the path name where the training images are saved
       exp_type:       (String)
                       'control',
                       'peripher_lowpassfilter',
                       'peripher_highpassfilter'
                       'zentral_lowpassfilter',
                       'zentral_highpassfilter'
       search_type:     (String)
                       'target_not_present'
                        'expectedlocation'
                        'unexpectedlocation'

       bw=30:           brandwith for the KDE
       test_ratio: splitting ration
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
                                               search_type)
    create_image_masks(images_withFixations,
                       output_path,
                       bw,
                       test_ratio)


def main():
    create_masks_images_from_data(dir_fixations,
                                  dir_images,
                                  output_path,
                                  exp_type,
                                  search_type,
                                  bw,
                                  test_ratio,
                                  seed)


if __name__ == '__main__':
    main()