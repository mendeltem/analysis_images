#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:26:50 2020

@author: mendel
"""
import numpy as np
import pandas as pd
import tensorflow as tf


import os
import seaborn as sns
import matplotlib.pyplot as plt

from keras.utils import Progbar
from keras import Sequential
from keras.layers import Dense
from keras.utils.generic_utils import to_list
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import shuffle
import random
random.seed = 7

DEBUG = 0

from keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer




from modules.dataloader_featureklassification import Data_Loader


from modules.helper_function import rounding
import warnings
warnings.filterwarnings("ignore", category = UserWarning)
warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)

from modules.models import  simple_classifier

import matplotlib.pyplot as plt

import re

import feather


memory_color_path = '/mnt/data/DATA/dataset/FEATURES/saved/exception/memory/color'
experiment_path = '/mnt/data/DATA/blickdaten/finalresulttab_funcsac_SFC_memory.dat'
#
#layer_filter = "^1_|^2_|^3_|^4_|^5_|^6_|^7_|^8_|^9_|^10_|^11_"

layer_filter = "^1_|^2_"

filter1 =""
filter2 =""
batch_size = 40
epochs = 100
train_test_split_ratio = 0.3
showpath = 0


label_cols = [""]

# cols= [ 'colorimages', 'trialstart',
# 'trialend', 'stimulusstart', 'stimulusend', 'imageid', 'responsegiven',
# 'responsecorrect', 'fixno', 'fixonset',
# 'fixoffset', 'fixdur', 'fixposx', 'fixposy', 'fixinvalid', 'sacno',
# 'saconset', 'sacoffset', 'sacdur', 'velpeak', 'distance', 'angle1',
# 'amplitude', 'sacinvalid', 'wobblehandled']


drop_cols = ["stimulusstart", 'wobblehandled', 'sacinvalid']


loader = Data_Loader(memory_color_path,
                     experiment_path, 
                     layer_filter,
                     filter1,
                     filter2,
                     batch_size,
                     train_test_split_ratio,
                     drop_cols= [],        
                     showpath = 0
                     )


part_data = pd.read_feather(loader.filtered_train_paths[0])





input_shape = loader.get_shapes()

print("input shape: ", input_shape)


classifier = simple_classifier(input_shape)

classifier.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.00005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
      
classifier.summary()

training_step = loader.get_tr_step()
print("training shape: ", training_step)


test_step  =  loader.get_test_step()
print("test shape: ", test_step)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

train_id = np.random.randint(1000)


#Logger
log = open("trainlogs/train_id_" + str(train_id) + ".log", "w+")
log.write("Traines Epochs: " +str(epochs) +"\n")
log.write("experiment filter1: " +filter1 +"\n")
log.write("experiment filter2: " +filter2 +"\n")
log.write("batch_size: " +str(batch_size) +"\n")
log.write("layer_filter Epochs: " +layer_filter +"\n")

history_train_loss = []
history_test_loss = []

history_train_acc = []
history_test_acc = []
  
for epoch in range(epochs):
  print("Epoch ", epoch+1)
  
  train_generator  = loader.load_train_data()
  
#  test_generator   = loader.load_test_data()
  
  metrics_collect = []
  mean_train_loss = []

  list_train_mean_acc  = []
  list_test_mean_acc   = []
  
  list_train_mean_loss  = []
  list_test_mean_loss   = []  
  
  max_test_accuracy = 0
  
  
  progbar_train = Progbar(training_step)
  
  log.write("Epoch: " +str(epoch) +"\n")
  
  for i in range(training_step):
    
    X_train, y_train = next(train_generator)
     
    print("\nTrain shape: ", X_train.shape)
  
    progbar_train.add(1)
  
    if i == 0 and epoch == 1:
      #imputer trained only on first iteration
      imp.fit(X_train)
      
    #impute missing data
    X_train_imp =imp.fit_transform(X_train)
    
    #normalize
    X_train = preprocessing.normalize(X_train_imp)
   
      
#    X_train, y_train = shuffle(X_train_normal, y_train )
#      
      
    train_metrics = classifier.train_on_batch(X_train,
                         y_train
                         )
    
    
    print("metrics: ", train_metrics)
    
#    metric = classifier.test_on_batch(X_train, y_train)

    list_train_mean_acc.append(train_metrics[1])
    
    list_train_mean_loss.append(train_metrics[0])
    
#    
#  progbar_test = Progbar(test_step)  
#    
#  for i in range(test_step):
#    
#    X_test, y_test = next(test_generator)
#     
#    print("\nTest shape: ", X_test.shape)
#  
#    progbar_test.add(1)
#      
#    #impute missing data
#    X_test_imp =imp.fit_transform(X_test)
#    
#    #normalize
#    X_test_normal = preprocessing.normalize(X_test_imp)
##    #shuffle
##    X_test, y_test = shuffle(X_test_normal, y_test )
#      
#    test_metrics = classifier.test_on_batch(X_test, y_test)
#
#    list_test_mean_acc.append(test_metrics[1])
#    
#    list_test_mean_loss.append(test_metrics[0])
    
#  if max_test_accuracy < np.mean(list_test_mean_acc):
#    max_test_accuracy = np.mean(list_test_mean_acc)
    
#    print("save : ", max_test_accuracy)
#
#    classifier.save('models/model_epoch' + str(epoch) + "_test_acc_" + str(max_test_accuracy ) + "_id_ " + str(train_id)  )
#    
#    
#  log_train = "train accuracy: " +  str(np.mean(list_train_mean_acc)) + "\n" + \
#              "train loss: "     +  str(np.mean(list_train_mean_loss)) + "\n" + \
#              "test accuracy: " + str(np.mean(list_test_mean_acc)) + "\n" + \
#              "test loss: " + str(np.mean(list_test_mean_loss)) + "\n"
#              
              
#  classifier.fit(x=X_train, 
#      y=y_train,
#      batch_size=256,
#      epochs=100,
#      verbose=1,
#      callbacks=None, 
#      validation_split=0.1,
#      validation_data =(X_test, y_test)
#      )
#  
#  
#               
#  print(log_train)  
#  
#  log.write(log_train)
#  
#  history_train_acc.append(np.mean(list_train_mean_acc))
#  history_test_acc.append(np.mean(list_test_mean_acc))
#  
#  history_train_loss.append(np.mean(list_train_mean_loss))
#  history_test_loss.append(np.mean(list_test_mean_loss))  
#  
#log.close()
#
#
#
#plt.plot(history_train_acc)
#plt.plot(history_test_acc)
#plt.plot(history_train_loss)
#plt.plot(history_test_loss)
#
#plt.show()

#  
#  
#len(loader.get_filtered_test_paths())
#  
  
    
    
    
      
      
      
#      perdict_train = classifier.predict(X_train)
#      
#      y_train_shaped = [np.argmax(p) + 1 for p in y_train]
#      
#      
#      prediction_train_shaped = [np.argmax(p) + 1 for p in perdict_train]
#      
#      
#      score_test = classifier.evaluate(X_test, y_test, batch_size=32)
#      
#      
#      perdict_test = classifier.predict(X_test)
#
#
#      y_test_shaped = [np.argmax(p) + 1 for p in y_test]
#      
#      
#      prediction_test_shaped = [np.argmax(p) + 1 for p in perdict_test]
#      
#      
#      prediction_train_shaped[:10]
#      
#      
#      train_score = classifier.evaluate(X_train, y_train, batch_size=64)
#
#      X_test, y_test    =   shuffle(X_test, y_test, random_state=0)
      
#      
#  
#    
#  test_log  = "Test : " + str(test_mean_acc)+" \n"
#    
#  print(test_log)
##
##  log.write(test_log)
#  
##close writing the metrics in log  
#log.close()
#
##
##
###
#loader_test = Data_Loader(memory_color_path,
#                     layer_filter,
#                     filter1,
#                     filter2,
#                     batch_size,
#                     train_test_split_ratio = 0.1
#                     )
##
##
#gen = loader_test.load_train_dataframe()
#
#
#X_train, y_train = next(gen)
##
##
##X_train.shape
##
##y_train.shape
##
##
#
#



#


