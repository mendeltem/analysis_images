#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 12:09:38 2018

@author: panda
"""
#import pickle
import numpy as np
#import re
import sys
import os
from numbers import Number
from collections import Set, Mapping, deque
#import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from skimage.transform import resize
import re
import seaborn as sns
import imageio
from sklearn.neighbors.kde import KernelDensity


#monitor
my_dpi = 90
#path length for ploting kde images
pad_length = 27
#widh and height
#w = 1024+pad_length
#h = 768+pad_length
w = 1024
h = 768


def union(lst1, lst2): 
    final_list = lst1 + lst2 
    return final_list 

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

def memory_usage(df):
  """ Check the memory usage from a dataframe
  """
  return(round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2))

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

    width_scale = width_shape/layer_shape[2]
    #print("width_scale",width_scale)
    hight_scale = hight_shape/layer_shape[1]
    #print("hight_scale",hight_scale)
    xs = int(fix_x / width_scale)
    ys = int(fix_y / hight_scale)

    #scaled grade
    gx = int(fovea / width_scale)
    gy = int(fovea / hight_scale)

    solutions = []
    for i in range(layer_shape[0]):
        solutions.append(layer_matrix[i,xs-gx:xs+gx+1,ys-gy:ys+gy+1].mean())


    #print("type",type(solutions))
    return solutions


def get_spatial_activation(features, eye_movements = [], width_shape = (), hight_shape = ()):
    pass


    #for i in range(len(eye_movements)):
    #    for z in range(len(eye_movements[i])):
    #       print(eye_movements.iloc[i,z])

    #def get_spatial_activation(picture_layers,vec = [],width_shape, hight_shape):
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

    image_count = len(features)
    #for each image there is a list of eyemovement
    for i in range(image_count):
        activations = []
        #number of Eyemovements
        for u in range(len(eye_movements.iloc[i,1])):
            temp = []
            for z in range(11):
                temp += scaling_point_picture_to_layer(features[i][z],
                                                       eye_movements.iloc[i,0][z],
                                                       eye_movements.iloc[i,1][z],
                                                       width_shape,
                                                       hight_shape)
            activations.append(temp)
        list_of_activations.append(activations)
    return list_of_activations


try: # Python 2
    zero_depth_bases = (basestring, Number, xrange, bytearray)
    iteritems = 'iteritems'
except NameError: # Python 3
    zero_depth_bases = (str, bytes, Number, range, bytearray)
    iteritems = 'items'

def getsize(obj_0):
    """Recursively iterate to sum size of object & members."""
    _seen_ids = set()
    def inner(obj):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, zero_depth_bases):
            pass # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, iteritems):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, iteritems)())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size
    return inner(obj_0)

def get_img_paths(start_dir, extensions = ['png','jpeg','jpg','pneg','peng']):
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

def get_dat_paths(start_dir, extensions = ['dat']):
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

def get_feature_paths(start_dir, extensions = ['h5']):
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

def get_csv_paths(start_dir, extensions = ['csv']):
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
                                ((data["masktype"] == 0) | (data["masktype"] == 1) | (data["masktype"] == 2)) &
                                ((data["maskregion"]  == 1 ) | (data["maskregion"]  == 0 ))
                                ,
                                :]
    return selected_data



#
#def plot_training_history(history):
#
#
#    keys_list = [keys for keys in history.history.keys()]
#
#    # Get the classification accuracy and loss-value
#    # for the training-set.
#    acc = history.history[keys_list[3]]
#    loss = history.history[keys_list[2]]
#
#    # Get it for the validation-set (we only use the test-set).
#    val_acc = history.history[keys_list[1]]
#    val_loss = history.history[keys_list[0]]
#
#    # Plot the accuracy and loss-values for the training-set.
#    plt.plot(acc, linestyle='-', color='b', label='Training '+ str(keys_list[3]) )
#    plt.plot(loss, 'o', color='b', label='Training Loss')
#
#    # Plot it for the test-set.
#    plt.plot(val_acc, linestyle='--', color='r', label='Test Acc'+str(keys_list[1]))
#    plt.plot(val_loss, 'o', color='r', label='Test Loss')
#
#    # Plot title and legend.
#    plt.title('Training and Test Accuracy')
#    plt.legend()
#
#    plt.savefig("figures/loss")
#
#    # Ensure the plot shows correctly.
#    plt.show()


def plot_training_history(history, save_dir="figures/new/", plotname="history.jpg" ):


    keys_list = [keys for keys in history.keys()]

    # Get the classification accuracy and loss-value
    # for the training-set.
    acc = history[keys_list[3]]
    loss = history[keys_list[2]]

    # Get it for the validation-set (we only use the test-set).
    val_acc = history[keys_list[1]]
    val_loss = history[keys_list[0]]

    plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)

    # Plot the accuracy and loss-values for the training-set.
    plt.loglog(acc, linestyle='-', color='b', label='Train DLoss: ' +str(round(acc[-1], 3)))
    plt.loglog(loss, 'o', color='b', label='Train Loss: '+str(round(loss[-1], 3)))

    # Plot it for the test-set.
    plt.loglog(val_acc, linestyle='--', color='r', label='Test DLoss: '+str(round(val_acc[-1], 3)))
    plt.loglog(val_loss, 'o', color='r', label='Test Loss: '+str(round(val_loss[-1], 3)))

    # Plot title and legend.
    plt.title('Training and Test Loss')
    plt.legend()

    plt.savefig(save_dir+plotname)

    # Ensure the plot shows correctly.
    plt.show()



def plot_history(history):

  #keys_list = [keys for keys in history.keys()]

  # Get the classification accuracy and loss-value
  # for the training-set.
  #acc = history[keys_list[3]]
  #loss = history[keys_list[2]]

  # Get it for the validation-set (we only use the test-set).
  #val_acc = history[keys_list[1]]
  #val_loss = history[keys_list[0]]

  loss_list = [s for s in history.keys()     if 'loss' in s and 'val' not in s]
  val_loss_list = [s for s in history.keys() if 'loss' in s and 'val' in s]

  if len(loss_list) == 0:
      print('Loss is missing in history')
      return

  ## As loss always exists
  epochs = range(1,len(history[loss_list[0]]) + 1)

  ## Loss
  plt.figure(1)
  for l in loss_list:
    plt.plot(epochs, history[l], 'b', label='Training '+str(history[l])+ 'loss (' + str(str(format(history[l][-1],'.5f'))+')'))

  for l in val_loss_list:
      plt.plot(epochs, history[l], 'g', label='Validation loss (' + str(str(format(history[l][-1],'.5f'))+')'))

  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  ## Accuracy
#  plt.figure(2)
#  for l in acc_list:
#      plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
#  for l in val_acc_list:
#      plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

#  plt.title('Accuracy')
#  plt.xlabel('Epochs')
#  plt.ylabel('Accuracy')
#  plt.legend()
#  plt.show()





#import numpy as np
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import average_precision_score
#from sklearn.preprocessing import normalize
#import seaborn as sns
#
#import matplotlib.pyplot as plt





w = 1024
h = 768
my_dpi = 90
pad_length = 27


def evaluate(oritinal,pred,true,threshhold=0.0, save_dir="",history="", index=1):
  """Evaluate a matrix
  with accuracy precision and recall

  input:
    orignal input image
    prediction out from model
    true label mask
    mind: tollerance

  """
  plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
  oritinal = oritinal[index]

  pred = pred[index]
  true = true[index]

  y_pred = np.squeeze(pred)
  y_true = np.squeeze(true)


#
#  y_true.shape[0] * y_true.shape[1]
#
#  plt.imshow(y_pred)
#  plt.imshow(y_true)
##
#  np.sum(true_positiv)
#  np.sum(true_negativ)
#  np.sum(flase_positiv)
#  np.sum(false_negativ)
  #y_pred = normalize(y_pred, axis=1, norm='l1')
  #y_true = normalize(y_true, axis=1, norm='l1')

#  #true positiv
#  true_positiv_mask = y_pred * y_true
#  true_positiv = np.array(list(map(lambda x : x > threshhold, true_positiv_mask)))
#  #true negativ
#  true_negativ_mask = (1 - y_pred) * np.logical_not(y_true).astype(int)
#  true_negativ = np.array(list(map(lambda x : x > threshhold, true_negativ_mask)))
#  #false positiv
#  flase_positiv_mask = y_pred * np.logical_not(y_true).astype(int)
#  flase_positiv = np.array(list(map(lambda x : x > threshhold, flase_positiv_mask)))
#  #false negativ
#  false_negativ_mask = (1 - y_pred) * y_true
#  false_negativ = np.array(list(map(lambda x : x > threshhold, false_negativ_mask)))
#



  #true positiv
  true_positiv_mask = np.multiply( y_pred , y_true)
  true_positiv = np.array(list(map(lambda x : x > threshhold, true_positiv_mask)))
  #true negativ
  true_negativ_mask = np.multiply((1 - y_pred) , (1-y_true)    )
  true_negativ = np.array(list(map(lambda x : x > threshhold, true_negativ_mask)))
  #false positiv
  flase_positiv_mask = np.multiply(y_pred , np.logical_not(y_true).astype(int))
  flase_positiv = np.array(list(map(lambda x : x > threshhold, flase_positiv_mask)))
  #false negativ
  false_negativ_mask  = np.multiply(np.where(y_pred <threshhold, 1, 0),y_true)
  false_negativ = np.array(list(map(lambda x : x > threshhold, false_negativ_mask)))

  #accuracy: TP+TN/TP+TN+FP+FN
  accuracy = (np.sum(true_positiv) + np.sum(true_negativ))/ (np.sum(true_positiv) + np.sum(true_negativ) + np.sum(flase_positiv)+ np.sum(false_negativ))

  #accuracy = np.sum(true_positiv + true_negativ)/ np.sum(true_positiv + true_negativ + flase_positiv+ false_negativ)

  print("accuracy:    ", np.round(accuracy, decimals=2))
  #precision TP/TP+FP
  precision = np.sum(true_positiv) / (np.sum(true_positiv) + np.sum(flase_positiv))
  print("precision:   ", np.round(precision, decimals =2))
  #recall TP /TP+FN
  recall = np.sum(true_positiv) / np.sum( true_positiv  +false_negativ)
  print("recall:      ", np.round(recall, decimals =2))



  fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15,15))
  plt.title("Train")
  plt.subplot(3, 3, 1)
  plt.title('original')
  plt.imshow(oritinal)
  plt.subplot(3, 3, 2)
  plt.title('TrueLabel')
  plt.imshow(y_true)
  plt.subplot(3, 3, 3)
  plt.title('prediction')
  plt.imshow(y_pred)
  plt.subplot(3, 3, 4)
  plt.title('true_positiv_mask')
  plt.imshow(true_positiv_mask)
  plt.subplot(3, 3, 5)
  plt.title('true_negativ_mask')
  plt.imshow(true_negativ_mask)
  plt.subplot(3, 3, 6)
  plt.title('flase_positiv')
  plt.imshow(flase_positiv)
  plt.subplot(3, 3, 7)
  plt.title('false_negativ')
  plt.imshow(false_negativ)
  plt.subplot(3, 3, 8)
  plt.title('Training and Test Loss')
  plt.loglog(history["val_loss"], linestyle='-', color='r', label='Val Loss: ' +str(round(history["val_loss"][-1], 3)))
  plt.loglog(history["loss"], linestyle='-', color='b', label='Loss: ' +str(round(history["loss"][-1], 3)))
  plt.legend()

#  plt.subplot(3, 3, 9)
#  plt.scatter([accuracy,precision,recall ])

  plt.savefig(save_dir+"false_negativ.jpg")

  plt.show()


def get_eval(prediction,label,threshhold):

  y_pred = np.squeeze(prediction)
  y_true = np.squeeze(label)

  #y_pred = normalize(y_pred, axis=1, norm='l1')
  #y_true = normalize(y_true, axis=1, norm='l1')

  print(y_pred.shape)
  print(y_true.shape)

  #true positiv
  true_positiv_mask = np.multiply( y_pred , y_true)
  true_positiv = np.array(list(map(lambda x : x > threshhold, true_positiv_mask)))
  #true negativ
  true_negativ_mask = np.multiply((1 - y_pred) , np.logical_not(y_true).astype(int))
  true_negativ = np.array(list(map(lambda x : x > threshhold, true_negativ_mask)))
  #false positiv
  flase_positiv_mask = np.multiply(y_pred , np.logical_not(y_true).astype(int))
  flase_positiv = np.array(list(map(lambda x : x > threshhold, flase_positiv_mask)))
  #false negativ
  false_negativ_mask = np.multiply( (1 - y_pred) , y_true)




  false_negativ = np.array(list(map(lambda x : x > threshhold, false_negativ_mask)))

  #accuracy: TP+TN/TP+TN+FP+FN
  accuracy = (np.sum(true_positiv) + np.sum(true_negativ))/  (np.sum(true_positiv) + np.sum(true_negativ) + np.sum(flase_positiv)+ np.sum(false_negativ))
  print("accuracy:    ",np.round(accuracy, decimals=2))
  #precision TP/TP+FP
  precision = np.sum(true_positiv) / (np.sum(true_positiv) + np.sum(flase_positiv))
  print("precision:   ",np.round(precision, decimals =2))
  #recall TP /TP+FN
  recall = np.sum(true_positiv) / (np.sum(true_positiv)  + np.sum(false_negativ))
  print("recall:      ",np.round(recall, decimals =2))

  return accuracy,precision,recall


def evaluate_model(pred, labels, threshhold):
  """get the mean accuracy precision and recall from the model

  input: list of predictions
         list of labels
         threshhold

         returns: mean accuracy, precision and recall


  """

  accuracy_list=[]
  precision_list=[]
  recall_list=[]

  for prediction,label in zip(pred,labels):

    accuracy,precision,recall = get_eval(prediction,label, threshhold)

    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)

  return accuracy_list,precision_list,recall_list



def create_dir(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)


def intersection_of_2_list(list1, list2):
    return [value for value in list1 if value in list2]   
        

def get_keyname_from_true_dict(dictiony):
    list_of_expected = []
    for i, info in enumerate(list(dictiony.keys())):
        if dictiony[info]:
            list_of_expected.append(info)
    return list_of_expected     


def get_word_contains_in_list(list1, word):
    return [i for i in list1 if word in i] 


def get_keynames_contain_keyword(data, keyword): 
    return (([key for i, key in enumerate(\
                         list(data.keys())) if keyword in key]))  
            
            
def get_first_number_from_path(path):
        return int(re.search(r'\d+', 
          os.path.basename( path)).group())    


        
def create_kde(x,y ,KERNEL, BANDWITH,cmap,SHADE, N_LEVELS, CUMULATIVE, size):
  

    fig = sns.kdeplot(
                x,
                y, 
                kernel = KERNEL,
                bw= BANDWITH,
                cmap = cmap,
                shade=SHADE, 
                n_levels = N_LEVELS,
                legend= False,
                cumulative= CUMULATIVE
                )
         
    plt.close()
    fig.axes.get_yaxis().set_visible(False)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.invert_yaxis()
   
    figure = fig.get_figure()
    figure.tight_layout(pad=0)
    figure.canvas.draw()
    mask = np.fromstring(
            figure.canvas.tostring_rgb(),dtype=np.uint8,sep='')
    mask = mask.reshape(
            figure.canvas.get_width_height()[::-1] + (3,))
    return  resize(mask,
                   size, 
                   preserve_range = True)
    
def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs): 
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins, 
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)
  
        
def kde2D_plot(x,y, bw,size):
    """ploting 2D kernel density estimation 
    
    """
    xx, yy, zz = kde2D(x, y, bw)
    
    fig = plt.pcolormesh(xx, yy, zz)
    plt.close()
   
    fig.axes.get_yaxis().set_visible(False)
    fig.axes.get_xaxis().set_visible(False)
    
    fig.axes.invert_yaxis()
    figure = fig.get_figure()
    figure.tight_layout(pad=0)
    figure.canvas.draw()
    mask = np.fromstring(figure.canvas.tostring_rgb(),dtype=np.uint8,sep='')
    mask = mask.reshape( figure.canvas.get_width_height()[::-1] + (3,))
    
    mask = resize(mask,size,preserve_range = True) 
    
    return  mask
      
    
            
def save_image(image, path):
    
    image = image.astype(np.float64) / np.max(image)
    image = (image * 255).astype(np.uint8)
    
    
    imageio.imwrite(
            path, 
            image)          
    
    
    
            