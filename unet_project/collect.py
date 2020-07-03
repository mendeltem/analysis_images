#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:26:50 2020

@author: mendel
"""
import numpy as np
import pandas as pd
import tensorflow as tf

tf.enable_eager_execution()

import os
import feather
import sys
import seaborn as sns
import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from pympler import asizeof

import datetime
import re
import random
random.seed = 7

from keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


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
    
  else: return 0


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

 
class Data_Loader:
  """my new class to provide
  
  """
  def __init__(self, 
               path,
               train_test_split_ratio = 0.1):
    
      path = memory_color_path
    
      self.path = path
      self.paths =  get_file_paths(path)
      
      random.shuffle(self.paths)

      train_idx = int(len(self.paths) * train_test_split_ratio)
      
      self.trainpaths =  self.paths[train_idx:]
      self.testpaths  =  self.paths[:train_idx]
      

  def get_filtered_train_paths(self, filter1, filter2):
    return filter_paths( self.trainpaths, filter1, filter2)
  
  
  def get_filtered_test_paths(self, filter1, filter2):
    return filter_paths( self.testpaths, filter1, filter2)  
  

  def get_tr_step(self, filter1, filter2, batch_size):
    return int(len(filter_paths( self.trainpaths, filter1, filter2)) / batch_size)
  
  
  def get_test_step(self, filter1, filter2, batch_size):
    return int(len(filter_paths( self.testpaths, filter1, filter2)) / batch_size)
  
  
  def get_shapes(self,
                layer_filter,
                filter1,
                filter2
                ):

    f_apths = self.get_filtered_train_paths( filter1, filter2)


    t  = 0

      
    train_data = pd.DataFrame()

       
    part_data = pd.read_feather(f_apths[t])
  
    experiment_idx = [i for i,col_name in enumerate(part_data.columns)\
                 if not re.search("_|index", col_name)]
    
    input_idx = [i for i,col_name in enumerate(part_data.columns)\
                 if re.search(layer_filter, col_name)]
    
    index_c = np.concatenate( (experiment_idx, input_idx) )

    train_data = pd.concat([train_data, part_data.iloc[:,index_c]], 
                        axis=0,
                        ignore_index=True)   
    


    X_train = train_data.drop(["masktype","maskregion"], axis =1)
    X_train = X_train.values
 
    
    return X_train.shape[1]
      
      
  def load_train_data(self,
                layer_filter,
                filter1,
                filter2,
                batch_size,
                shuffle = 1
                ):

    f_apths = self.get_filtered_train_paths( filter1, filter2)
    
  
    trainin_step = self.get_tr_step(filter1, filter2, batch_size)
  
    t  = 0
    for i in range(trainin_step):
      
      train_data = pd.DataFrame()
  
      for z in range(batch_size):
  
        
        #print(z,": paths is loading:",f_apths[t])
           
        part_data = pd.read_feather(f_apths[t])
      
        experiment_idx = [i for i,col_name in enumerate(part_data.columns)\
                     if not re.search("_|index", col_name)]
        
        input_idx = [i for i,col_name in enumerate(part_data.columns)\
                     if re.search(layer_filter, col_name)]
        
        index_c = np.concatenate( (experiment_idx, input_idx) )
    
        train_data = pd.concat([train_data, part_data.iloc[:,index_c]], 
                            axis=0,
                            ignore_index=True)   
        
        t += 1
          
        
      train_labels = train_data.loc[:,["masktype","maskregion"]].values
      train_y = np.array(list(map(filter_experiment,train_labels)))
      
      X_train = train_data.drop(["masktype","maskregion"], axis =1)
      X_train = X_train.values
 
      
      yield (X_train, train_y) 
        
  def load_test_data(self,
                layer_filter,
                filter1,
                filter2,
                batch_size,
                shuffle = 1
                ):

    f_apths = self.get_filtered_test_paths( filter1, filter2)
    
  
    self.test_step = self.get_test_step(filter1, filter2, batch_size)
     
    t  = 0
    for i in range(self.test_step):
      
      test_data = pd.DataFrame()
  
      for z in range(batch_size):
  
        #print(z,": paths is loading:",f_apths[t])
           
        part_data = pd.read_feather(f_apths[t])
      
        experiment_idx = [i for i,col_name in enumerate(part_data.columns)\
                     if not re.search("_|index", col_name)]
        
        input_idx = [i for i,col_name in enumerate(part_data.columns)\
                     if re.search(layer_filter, col_name)]
        
        index_c = np.concatenate( (experiment_idx, input_idx) )
    
        test_data = pd.concat([test_data, part_data.iloc[:,index_c]], 
                            axis=0,
                            ignore_index=True)   
        
        t += 1
          
        
      test_labels = test_data.loc[:,["masktype","maskregion"]].values
      test_y = np.array(list(map(filter_experiment,test_labels)))
      
      X_test = test_data.drop(["masktype","maskregion"], axis =1)
      X_test = X_test.values
 
      
      yield (X_test, test_y) 
#loader = Data_Loader(memory_color_path
#                     )
#  
#
#loader.get_filtered_paths( 
#    filter1 ="normal",
#    filter2 ="color")
#

##
#loader.load_data(
#                layer_filter,
#                filter1,
#                filter2,
#                batch_size
#                )
#
      
def create_model(input_shape):
  return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(input_shape,)),
    tf.keras.layers.Dense(512, activation='relu', dtype='float32'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu', dtype='float32'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='softmax')
  ])


memory_color_path = '/mnt/data/DATA/dataset/FEATURES/saved/exception/memory/color'

layer_filter = "^1_|^2_|^3_|^4_|^5_|^6_|^7_|^8_|^9_|^10_|^11_"

layer_filter = "^1_"

filter1 ="normal"
filter2 ="color"
batch_size =  10
test_batchsize = 10
epochs = 100

loader = Data_Loader(memory_color_path, train_test_split_ratio = 0.1)


input_shape = loader.get_shapes(  layer_filter,
                    filter1,
                    filter2
            )

training_step = loader.get_tr_step( 
    filter1 ="normal",
    filter2 ="color",
    batch_size = batch_size 
    )

test_step = loader.get_test_step( 
    filter1 ="normal",
    filter2 ="color",
    batch_size = test_batchsize
    )

imp = SimpleImputer(missing_values=np.nan, strategy='mean')


model = create_model(input_shape)

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.00005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1)

for epoch in range(epochs):
  print("Epoch ", epoch+1)
  
  train_generator  = loader.load_train_data(
                      layer_filter,
                      filter1,
                      filter2,
                      batch_size 
                      )

  metrics_collect = []

  for i in range(training_step):
    
    X_train, y_train = next(train_generator)
    
    if X_train.shape[0]:
    
      if i == 0:
        imp.fit(X_train)
      
      #impute missing data
      X_train_imp =imp.fit_transform(X_train)
      
      #normalize
      X_train_normal = preprocessing.normalize(X_train_imp)
      
      np.random.shuffle(X_train_normal)
      np.random.shuffle(y_train)
      
      metrics = model.train_on_batch(X_train_normal,
                           y_train
                           )
      
      metrics_collect.append(metrics)
      
      
  metrics_mean = np.mean(np.array(metrics_collect), axis= 0)  
      

  print("Train: "+str(model.metrics_names[0])+" "+  str(metrics_mean[0])+"  "+\
        str(model.metrics_names[1])+" "+  str(metrics_mean[1]))


  test_generator = loader.load_test_data(
                        layer_filter,
                        filter1,
                        filter2,
                        batch_size = test_batchsize
                        )
  loss_collection = []
  acc_collection = []
  
  for i in range(test_step):
    
    X_test, y_test = next(test_generator)
    
    
    if X_test.shape[0]:
    
      if i == 0:
        imp.fit(X_test)
      
      #impute missing data
      X_test_imp =imp.fit_transform(X_test)
      
      #normalize
      X_test_normal = preprocessing.normalize(X_test_imp)
      
#      np.random.shuffle(X_test_normal)
#      np.random.shuffle(y_test)
#           

      prediction =model.predict_on_batch(X_test_normal)
      
      prediction = np.array(prediction)
      
      prediction_array = np.array([round_prediction(np.argmax(p)) for p in prediction ])
      
      loss = np.sum((prediction_array - y_test)**2) / len(prediction_array)
      
      acc = accuracy(y_test, prediction_array)
        
      loss_collection.append(loss)
      
      acc_collection.append(acc)
    
    
  metrics_mean_loss = np.mean(np.array(loss_collection), axis= 0)   
  
  metrics_mean_acc = np.mean(np.array(acc_collection), axis= 0)   
    
  print("Test:  loss: "+ str(metrics_mean_loss )+"  acc:" +str(metrics_mean_acc))


