#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:26:50 2020

@author: mendel
"""
import numpy as np
import pandas as pd
import tensorflow as tf
#import timeout_decorator


import os
import seaborn as sns
import matplotlib.pyplot as plt

from keras.utils import Progbar
from keras import Sequential
from keras.layers import Dense
from keras.utils.generic_utils import to_list
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.utils import shuffle
import datetime
import re
import random
random.seed = 7

DEBUG = 0

from keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from modules.helper_function import get_file_paths, my_shuffle, filter_experiment 
from modules.helper_function import filter_paths, round_prediction, MyOneHotEncoder 
from modules.helper_function import filter_experiment_sparse


class Data_Loader:
  """my new class to provide the perceptron classificator
  It trains on batches 
  
  Round prediction arrays
  
  arguments:
       path  = dataset path
       filter1        = filter experiment enviroment
       filter2        = filter experiment enviroment
       batch_size     = eye movement data from images are loaded
       train_test_split_ratio = 0.1
    
  
  """
  def __init__(self, 
               files_path,
               experiment_path,
               layer_filter,
               filter1,
               filter2,
               batch_size,
               train_test_split_ratio,
               drop_cols= [],        
               showpath = 0
               ):
 
    
#      timeout_decorator.timeout(1)
      
      self.files_path = files_path
      self.paths =  get_file_paths(files_path)
      random.shuffle(self.paths)
    
      experiment_path = experiment_path
      experiment_table  = pd.read_table(experiment_path)  
      self.subject_list = np.unique(experiment_table['subject'])
      self.image_list   = np.unique(experiment_table['imageid'])
      
      
      self.subject_name_list =  ["subject_"+str(name) for i,name in enumerate(self.subject_list)]
      self.image_name_list =  ["image_"+str(name) for i,name in enumerate(self.image_list)]
      
      self.encoders = {}
      self.encoders['subject'] = MyOneHotEncoder(self.subject_list)
      self.encoders['imageid'] = MyOneHotEncoder(self.image_list)
      self.batch_size = batch_size
      self.showpath = showpath
      self.layer_filter = layer_filter

     
      train_idx = int(len(self.paths) * train_test_split_ratio)
      
      self.trainpaths =  self.paths[train_idx:]
      self.testpaths  =  self.paths[:train_idx]
      
      
      self.train_step = int(len(filter_paths( self.trainpaths,
                                             filter1, filter2)) / batch_size)
  
      self.test_step = int(len(filter_paths( self.testpaths,
                                            filter1, filter2)) / batch_size)
      
      self.filtered_train_paths = filter_paths( self.trainpaths, filter1,
                                               filter2)
      
      self.filtered_test_paths  = filter_paths( self.testpaths, filter1,
                                                filter2)  
      
      self.col_one_hot =  ['subject', 'imageid']
      
      
      self.cols = drop_cols
      
      
  def get_filtered_train_paths(self):
    return self.filtered_train_paths
  
  
  def get_filtered_test_paths(self):
    return self.filtered_test_paths
  

  def get_tr_step(self):
    return self.train_step
  
  
  def get_test_step(self):
    
    if self.test_step == 0:
      return 1
    else:
      return self.test_step
    

  def get_shapes(self ):
    """Get input shape for the network"""

    f_apths  = self.get_filtered_train_paths()
    train_data = pd.DataFrame()
    print(": paths is loading:",f_apths[0])
    part_data = pd.read_feather(f_apths[0])

    experiment_idx = [i for i,col_name in enumerate(part_data.columns)\
                 if not re.search("_|index", col_name)]
    
    if self.layer_filter:
      input_idx = [i for i,col_name in enumerate(part_data.columns)\
                   if re.search(self.layer_filter, col_name)]
    else:
      input_idx = [] 
        
    print("self.layer_filter ", self.layer_filter)  
    print("input_idx ", len(input_idx))  
    print("experiment_idx ", experiment_idx)  
    index_c = np.concatenate( (experiment_idx, input_idx) )

    print("index_c ", index_c)
    train_data = pd.concat([train_data, part_data.iloc[:,index_c]], 
                        axis=0,
                        ignore_index=True)   
    
    #Hot encoding
    encoded_subject_matrix = self.encoders['subject'].transform(train_data['subject'])
    encoded_image_matrix   = self.encoders['imageid'].transform(train_data['imageid'])
    
    hotencoded_subject_dataframe =  pd.DataFrame(encoded_subject_matrix, 
                                                 columns = self.subject_name_list)
    
    hotencoded_image_dataframe   =  pd.DataFrame(encoded_image_matrix, 
                                                 columns = self.image_name_list)   
    
    train_data = pd.concat([train_data, hotencoded_subject_dataframe], axis = 1)
    train_data = pd.concat([train_data, hotencoded_image_dataframe], axis = 1)
    
    #drop columns
    train_data = train_data.drop(["subject","imageid", "trialid",
                                  "trialno","masktype","maskregion"], axis =1)
    train_data = train_data.drop(self.cols, axis =1)
    
    return train_data.shape[1]

#      
  def load_train_data(self):

    f_apths = self.get_filtered_train_paths()
    len_paths = len(f_apths)
    trainin_step = self.get_tr_step()
    t = 0
  
    for i in range(trainin_step):
      
      train_data = pd.DataFrame()
      for z in range(self.batch_size):
        
        len_paths -= 1
        
        if len_paths == 0:
          break

        if self.showpath:
          print(z,": paths is loading:",f_apths[t])
           
        part_data = pd.read_feather(f_apths[t])
      
        experiment_idx = [i for i,col_name in enumerate(part_data.columns)\
                     if not re.search("_|index", col_name)]
        
        if self.layer_filter:
          input_idx = [i for i,col_name in enumerate(part_data.columns)\
                       if re.search(self.layer_filter, col_name)]
        else:
          input_idx = [] 
        
        index_c = np.concatenate( (experiment_idx, input_idx) )
    
        train_data = pd.concat([train_data, part_data.iloc[:,index_c]], 
                            axis=0,
                            ignore_index=True)   
        
        
        t += 1
        
      #Labels 
      train_labels = train_data.loc[:,["masktype","maskregion"]].values
      train_y = np.array(list(map(filter_experiment,train_labels)))
      
      #Datas

      #one hot encod
      encoded_subject_matrix = self.encoders['subject'].transform(train_data['subject'])
      encoded_image_matrix   = self.encoders['imageid'].transform(train_data['imageid'])
  
      hotencoded_subject_dataframe =  pd.DataFrame(encoded_subject_matrix, 
                                                   columns = self.subject_name_list)
      
      hotencoded_image_dataframe =  pd.DataFrame(encoded_image_matrix, 
                                                   columns = self.image_name_list)   
      
      train_data = pd.concat([train_data, hotencoded_subject_dataframe], axis = 1)
      train_data = pd.concat([train_data, hotencoded_image_dataframe], axis = 1)
      
      #drop columns
      train_data = train_data.drop(["subject","imageid", "trialid",
                                    "trialno","masktype","maskregion"], axis =1)
      train_data = train_data.drop(self.cols, axis =1)
      
      train_data = train_data.values
      
      train_data = train_data.astype('float32')
      
      train_y = train_y.astype('float32')
 
      yield (train_data, train_y) 
      
      
        
#  def load_test_data(self):
#
#    f_apths = self.get_filtered_test_paths()
#    
#    len_paths = len(f_apths)
#    
#    test_step = self.get_test_step()
#    
#    t = 0
#    
#    for i in range(test_step):
#      
#      test_data = pd.DataFrame()
#           
#      for z in range(self.batch_size):
#        
#        len_paths -= 1
#        
#        if len_paths == 0:
#          break
#        
#        try:
#          if self.showpath:
#            print(z,": paths is loading:",f_apths[t])
#             
#          part_data = pd.read_feather(f_apths[t])
#        
#          experiment_idx = [i for i,col_name in enumerate(part_data.columns)\
#                       if not re.search("_|index", col_name)]
#          
#          input_idx = [i for i,col_name in enumerate(part_data.columns)\
#                       if re.search(self.layer_filter, col_name)]
#          
#          index_c = np.concatenate( (experiment_idx, input_idx) )
#      
#          test_data = pd.concat([test_data, part_data.iloc[:,index_c]], 
#                              axis=0,
#                              ignore_index=True)   
#        except:
#          break
#        t += 1
#        
#      #Labels  
#      test_labels = test_data.loc[:,["masktype","maskregion"]].values
#      test_y = np.array(list(map(filter_experiment,test_labels)))
#      test_y = test_y.astype('float32')
#      #Datas
#
#      #one hot encod
#      encoded_subject_matrix = self.encoders['subject'].transform(test_data['subject'])
#      encoded_image_matrix   = self.encoders['imageid'].transform(test_data['imageid'])
#  
#      hotencoded_subject_dataframe =  pd.DataFrame(encoded_subject_matrix, 
#                                                   columns = self.subject_name_list)
#      
#      hotencoded_image_dataframe =  pd.DataFrame(encoded_image_matrix, 
#                                                   columns = self.image_name_list)   
#      
#      test_data = pd.concat([test_data, hotencoded_subject_dataframe], axis = 1)
#      test_data = pd.concat([test_data, hotencoded_image_dataframe], axis = 1)
#      
#      test_data = test_data.drop(["masktype","maskregion"], axis =1)
#      X_test = test_data.drop(self.cols, axis =1)
#
#      X_test = X_test.values
#      
#      X_test = X_test.astype('float32')
#      
#      yield (X_test, test_y) 
#      

#  def load_train_dataframe(self):
#
#    f_apths = self.get_filtered_train_paths()
#    
#    trainin_step = self.get_tr_step()
#    
#    
#    t = 0
#  
#    for i in range(trainin_step):
#      
#      train_data = pd.DataFrame()
#  
#      for z in range(self.batch_size):
#  
#        if self.showpath:
#          print(z,": paths is loading:",f_apths[t])
#           
#        part_data = pd.read_feather(f_apths[t])
#      
#        experiment_idx = [i for i,col_name in enumerate(part_data.columns)\
#                     if not re.search("_|index", col_name)]
#        
#        input_idx = [i for i,col_name in enumerate(part_data.columns)\
#                     if re.search(self.layer_filter, col_name)]
#        
#        index_c = np.concatenate( (experiment_idx, input_idx) )
#    
#        train_data = pd.concat([train_data, part_data.iloc[:,index_c]], 
#                            axis=0,
#                            ignore_index=True)   
#        
#        t += 1
#        
#      #Labels 
#      train_labels = train_data.loc[:,["masktype","maskregion"]].values
#      train_y = np.array(list(map(filter_experiment,train_labels)))
#      
#      
#      #Datas
#
#      #one hot encod
#      encoded_subject_matrix = self.encoders['subject'].transform(train_data['subject'])
#      encoded_image_matrix   = self.encoders['imageid'].transform(train_data['imageid'])
#  
#      hotencoded_subject_dataframe =  pd.DataFrame(encoded_subject_matrix, 
#                                                   columns = self.subject_name_list)
#      
#      hotencoded_image_dataframe =  pd.DataFrame(encoded_image_matrix, 
#                                                   columns = self.image_name_list)   
#      
#      train_data = pd.concat([train_data, hotencoded_subject_dataframe], axis = 1)
#      train_data = pd.concat([train_data, hotencoded_image_dataframe], axis = 1)
#      
#
#      yield (train_data, train_y, train_labels) 