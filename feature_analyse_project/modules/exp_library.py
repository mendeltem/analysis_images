#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 12:09:38 2018

@author: panda
"""
import pickle
import numpy as np
import re
import pandas as pd



def change_shape(array):
    """changing the shape (a,b,c)->(c,b,a)
    input array of an layer 3D dimenstional

    returns new reshaped array
    """
    new_array = []
    for i in range(array.shape[2]):
        col = []
        for y in range(array.shape[1]):
            row = []
            for z in range(array.shape[0]):
                row.append(array[z][y][i])
            col.append(row)
        new_array.append(col)

    return np.array(new_array)



def scaling_point_picture_to_layer(layer_matrix, fix_x, fix_y,  width_shape = (), hight_shape = ()):
    fovea = 30
    """scaling points from orinal picture to layer matrix
    Input:
    picture array
    layer_array
    x,y
    Returns the scaled x,y point
    """

    #print("original_shape:                 ", original_shape)
    layer_shape    = layer_matrix.shape
    #print("layer_shape # 1st 3rd switched:  ", layer_shape[0])

    width_scale = width_shape/layer_shape[1]
    #print("1 width_scale",width_scale)
    #print("1 layer_shape[2]",layer_shape[2])
    hight_scale = hight_shape/layer_shape[2]
    #print("2 hight_scale",hight_scale)
    #print("2 layer_shape[1]",layer_shape[1])
    xs = int(fix_x / width_scale)
    ys = int(fix_y / hight_scale)

    #scaled grade
    gx = int(fovea / width_scale)
    gy = int(fovea / hight_scale)

    solutions = []
    for i in range(layer_shape[0]):
        #print(xs-gx,xs+gx+1,ys-gy,ys+gy+1)
        solutions.append(layer_matrix[i,xs-gx:xs+gx+1,ys-gy:ys+gy+1].mean())


    #print("type",type(solutions))
    return solutions



def import_layerdata(file):
    """Get the Feature Data from the Inception Object

    Input: Inpeption Object File

    Returns:
        1. Features       List
        2. Path to the pictures
        3. Original FeatureDictionary
    """
    #load the inception file

    with open(file, 'rb') as f:
        feat_dict = pickle.load(f)
    #the number of images
    image_count = len(feat_dict["features"])
    #the number of layers used 11
    layer_count = len(feat_dict["features"][0])

    print("Image Count: ",image_count)
    print("Layer Count: ",layer_count)

    picture_layers = []
    picture_layer  = []
    image_labels   = []
    image_names    = []
    new            = []

    paths_of_pics  = []

    pattern = "/.*/"
    for i in range(image_count):
        temp = []
        #get all layers and reshape every layers
        image_names.append(re.sub(pattern," ",str(feat_dict["paths"][i])))
        image_labels.append(feat_dict["labels"][i])
        for y in range(layer_count):
             temp.append(
                 change_shape(
                         feat_dict["features"][i][y].reshape(
                                 feat_dict["features"][i][y][0].shape)))
        picture_layers.append(temp)
        #get only one layer and reshape every layers
        picture_layer.append(
                feat_dict["features"][i][0].reshape(
                        feat_dict["features"][i][0][0].shape))

        new.append(change_shape(picture_layer[i]))
        paths_of_pics.append(feat_dict["paths"][i])


    feat_dict.keys()

    return picture_layers, paths_of_pics, feat_dict



def get_spatial_activation(features, eye_movements = [], width_shape = (), hight_shape = ()):
    """Get the spatial activation from the Features where the eye movements
       Fixated

    Input:
    1. Original Picture  Picture
    2. Feature Array     List
    3. Foveal Area       Int
    4. Eye Movements     List


    Returns:
    1. The activated Areas as list of 10048

    """

    list_of_activations = []


    #image_count = print(len(eye_movements.index))

    image_count = len(features)
    #for each image there is a list of eyemovement


    #for i,n in enumerate(eye_movements.index):
    #    print(i,n)


    for i in range(image_count):
        activations = []
        #number of Eyemovements
        for u in range(len(eye_movements.iloc[i,1])):

            temp = []
            for z in range(11):
                temp += scaling_point_picture_to_layer(features[i][z],
                                                       eye_movements.iloc[i,0][u],
                                                       eye_movements.iloc[i,1][u],
                                                       width_shape,
                                                       hight_shape)
            activations.append(temp)
        list_of_activations.append(activations)
    return list_of_activations




def get_eyemovements(dataFrame):
    """Take a DataFrame from Eyemovement Experiment and returns the list
       eyemovemts. Preprozessing for

    Input: Selected Experment DataFrame

    Returns: DataFrame with lists of Eyemovement

    """

    ###create list of eyemovements
    list_of_ey_x = dataFrame.groupby("imageid")["fixposx"].apply(list)
    list_of_ey_y = dataFrame.groupby("imageid")["fixposy"].apply(list)
    list_of_ey_xy = pd.concat([list_of_ey_x,list_of_ey_y], axis = 1)

    return list_of_ey_xy




def data_selecting(data,color,masktype,maskregion,fixincalid):
    """choose the data associated to different experiment settings

    Arguments:
        data:

            ...

    Returns:
        DataFrame
    """
#masktype == 0 & maskregion == 0: Kontrollbedingung
    cleaned_data = data.loc[(data["colorimages"] == color) &
                                     (data["masktype"]    == masktype) &
                                     (data["maskregion"]  == maskregion) &
                                     (data["fixinvalid"]  == fixincalid) ,
                                     ['subject',
                                      'fixposx',
                                      "fixno",
                                      "fixposy",
                                      "imageid",
                                      "masktype",
                                      "maskregion",
                                      "fixinvalid",
                                      "colorimages"]]
    return cleaned_data








def get_spatial_activation_for_one_picture(features,
                                           eye_movements = [],
                                           width_shape = (),
                                           hight_shape = ()):
    """Get the spatial activation from the Features where the eye movements
       Fixated
    TODO
    Input:
    1. Original Picture  Picture
    2. Feature Array     List
    3. Foveal Area       Int
    4. Eye Movements     List


    Returns:
    1. The activated Areas as list of 10048

    """



    activations = []
    #number of Eyemovements
    for u in range(len(eye_movements.iloc[0,1])):

        temp = []
        for z in range(11):
            temp += scaling_point_picture_to_layer(features[z],
                                                   eye_movements.iloc[0,0][u],
                                                   eye_movements.iloc[0,1][u],
                                                   width_shape,
                                                   hight_shape)
        activations.append(temp)


    return activations