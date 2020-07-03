#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:25:31 2020

@author: mendel
"""

import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  
from sklearn.base import BaseEstimator, TransformerMixin
import re


def get_file_paths(start_dir, extensions = ['file']):
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



def my_shuffle(X,y):
  
  if len(y.shape) == 1:
    y = y.reshape(y.shape[0], 1)
    x_y_concate = np.concatenate((X, y ), axis = 1)
    np.random.shuffle(x_y_concate)
    
    return x_y_concate[:,:-1], x_y_concate[:,-1]
  else:
 
    x_y_conc = np.concatenate((X, y), axis = 1)
    np.random.shuffle(x_y_conc)
    return x_y_conc[:,:X.shape[1]] , x_y_conc[:,X.shape[1]:]
  
  
  
class MyOneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, all_possible_values):
        self.le = LabelEncoder()
        self.ohe = OneHotEncoder()
        self.ohe.fit(self.le.fit_transform(all_possible_values).reshape(-1,1))

    def transform(self, X, y=None):
        return self.ohe.transform(self.le.transform(X).reshape(-1,1)).toarray()  
      
      
      
def filter_experiment_sparse(y):
  """filter label
  
  arguments:
    y : array
    
  returns: 
  
    label: array
  """
  if y[0] == 0 and y[1] == 0: return 1
    #print("original: ",y)  
    
  if y[0] == 1 and y[1] == 2: return 2
    #print("zen_lp: ",y)
    
  if y[0] == 1 and y[1] == 1: return 3
    #print("per_lp: ",y) 
    
  if y[0] == 2 and y[1] == 1: return 4
    #print("per_hp: ",y)
    
  if y[0] == 2 and y[1] == 2: return 5
    #print("zen_hp: ",y)
    #print("zen_hp: ",y)
  else: return 0 


def filter_experiment(y):
  """filter label
  
  arguments:
    y : array
    
  returns: 
  
    label: array
  """

  if y[0] == 0 and y[1] == 0: return [1,0,0,0,0]
    #print("original: ",y)  
    
  if y[0] == 1 and y[1] == 2: return [0,1,0,0,0]
    #print("zen_lp: ",y)
    
  if y[0] == 1 and y[1] == 1: return [0,0,1,0,0]
    #print("per_lp: ",y) 
    
  if y[0] == 2 and y[1] == 1: return [0,0,0,1,0]
    #print("per_hp: ",y)
    
  if y[0] == 2 and y[1] == 2: return [0,0,0,0,1]
    #print("zen_hp: ",y)
  
  else: return [0,0,0,0,0]  
    
    


def round_prediction(y):
  """Round prediction arrays
  
  arguments:
    y : array
    
  returns: 
  
    label: array
  """
  
  if y == 0: return [1,0,0,0,0]
    #print("original: ",y)  
    
  if y == 1: return [0,1,0,0,0]
    #print("zen_lp: ",y)
    
  if y == 2: return [0,0,1,0,0]
    #print("per_lp: ",y) 
    
  if y == 3: return [0,0,0,1,0]
    #print("per_hp: ",y)
    
  if y == 4: return [0,0,0,0,1]
    #print("zen_hp: ",y)
    
  else: return None
    

def filter_paths(paths, 
                 filter_1="",
                 filter_2="",
                 filter_3 ="",
                 filter_4 = ""):  
  """Round prediction arrays
  
  arguments:
    y : array
    
  returns: 
  
    label: array
  """
  
  

  filtered_path = []
  
  for temp_path in paths:
    
    
    if(re.search(filter_1, temp_path) and \
       re.search(filter_2, temp_path) and \
       re.search(filter_3, temp_path) and \
       re.search(filter_4, temp_path)
       ):
      filtered_path.append(temp_path)
      
  return filtered_path    


def accuracy(prediction_array , y_test):
  
  accuracy_list = []

  for l,r in zip(prediction_array, y_test ):
     if (l == r).all():
       accuracy_list.append(1)
     else:
       accuracy_list.append(0)
       
  return np.sum(np.array(accuracy_list)) / len(accuracy_list)


def rounding(digits,threshold = 0.50):
    """rounds a list of float digits with ad threshhold"""
    if type(digits) == list  or  type(digits) == np.ndarray:
        return np.array(list(map(rounding, digits)))
    if type(digits)== np.float64 or type(digits)== np.float32:
        k = digits % 1
        f = digits - k

        if k >= threshold:
            return (f + 1)
        else:
            return f
    else:
        raise ValueError("Wrong Type") 
      