#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:56:13 2019

@author: mendel
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
import seaborn as sns

import matplotlib.pyplot as plt





w = 1024
h = 768
my_dpi = 90
pad_length = 27


def evaluate(oritinal,pred,true,threshhold, save_dir,history, index):
  """Evaluate a matrix 
  with accuracy precision and recall
  
  input:
    orignal input image
    prediction out from model
    true label mask
    mind: tollerance 
  
  """
  plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
  oritinal = oritinal
  
  y_pred = np.squeeze(pred)
  y_true = np.squeeze(true)
  
  #y_pred = normalize(y_pred, axis=1, norm='l1')
  #y_true = normalize(y_true, axis=1, norm='l1')
  
  
  print(y_pred.shape)
  print(y_true.shape)
  
  #true positiv
  true_positiv_mask = y_pred * y_true
  true_positiv = np.array(list(map(lambda x : x > threshhold, true_positiv_mask)))
  #true negativ
  true_negativ_mask = (1 - y_pred) * np.logical_not(y_true).astype(int)
  true_negativ = np.array(list(map(lambda x : x > threshhold, true_negativ_mask)))
  #false positiv
  flase_positiv_mask = y_pred * np.logical_not(y_true).astype(int)
  flase_positiv = np.array(list(map(lambda x : x > threshhold, flase_positiv_mask)))
  #false negativ
  false_negativ_mask = (1 - y_pred) * y_true
  false_negativ = np.array(list(map(lambda x : x > threshhold, false_negativ_mask)))
  
  #accuracy: TP+TN/TP+TN+FP+FN
  accuracy = (np.sum(true_positiv) + np.sum(true_negativ))/  (np.sum(true_positiv) + np.sum(true_negativ) + np.sum(flase_positiv)+ np.sum(false_negativ))
  print("accuracy:    ",accuracy)
  #precision TP/TP+FP
  precision = np.sum(true_positiv) / (np.sum(true_positiv) + np.sum(flase_positiv))
  print("precision:   ",precision)
  #recall TP /TP+FN
  recall = np.sum(true_positiv) / (np.sum(true_positiv)  + np.sum(false_negativ))
  print("recall:      ",recall)
  
  
  fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15,15))
  
  
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
  plt.subplot(3, 3, 9)
  plt.barh([accuracy,precision,recall], width = 0.75)
  
  
  
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
  true_positiv_mask = y_pred * y_true
  true_positiv = np.array(list(map(lambda x : x > threshhold, true_positiv_mask)))
  #true negativ
  true_negativ_mask = (1 - y_pred) * np.logical_not(y_true).astype(int)
  true_negativ = np.array(list(map(lambda x : x > threshhold, true_negativ_mask)))
  #false positiv
  flase_positiv_mask = y_pred * np.logical_not(y_true).astype(int)
  flase_positiv = np.array(list(map(lambda x : x > threshhold, flase_positiv_mask)))
  #false negativ
  false_negativ_mask = (1 - y_pred) * y_true
  false_negativ = np.array(list(map(lambda x : x > threshhold, false_negativ_mask)))
  
  #accuracy: TP+TN/TP+TN+FP+FN
  accuracy = (np.sum(true_positiv) + np.sum(true_negativ))/  (np.sum(true_positiv) + np.sum(true_negativ) + np.sum(flase_positiv)+ np.sum(false_negativ))
  print("accuracy:    ",accuracy)
  #precision TP/TP+FP
  precision = np.sum(true_positiv) / (np.sum(true_positiv) + np.sum(flase_positiv))
  print("precision:   ",precision)
  #recall TP /TP+FN
  recall = np.sum(true_positiv) / (np.sum(true_positiv)  + np.sum(false_negativ))
  print("recall:      ",recall)
  
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
    
    
    



#dice_loss(true_positiv, y_true)
#
#
#
#
#y_pred =np.array( [[0.38574776,0.08795848, 0.83927506],
#                  [0.21592768, 0.44453627, 0.10463644],
#                  [0.8793516,  0.65118235, 0.5184219 ]])
#
#
#y_test = np.array([[1,0,0],[1,0,0],[0,0,1]])
#
#
#
#
#Precision = y_pred * y_test
