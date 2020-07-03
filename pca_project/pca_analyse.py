#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:52:52 2019

@author: mendel
"""

import pickle
import pandas as pd
import numpy as np
import glob
from sklearn.decomposition import PCA
from sklearn import preprocessing
import os



import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#from modules.library import get_img_paths,get_feature_paths,memory_usage


def get_paths(start_dir, extensions = ['csv']):
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
  
def load_csv_files(path):
  
  paths = get_paths(path)
  df_inception = pd.DataFrame()

  for path in paths:
    temp_df = pd.read_csv(path)
    
    df_inception = pd.concat([df_inception, temp_df])
    
  return df_inception
  

datapath_inception = "/home/mendel/Documents/data/inception"


inception_feature_paths = load_csv_files(datapath_inception)


scaler = StandardScaler()

inception_values = df_inception.values

scaled_data_inception = StandardScaler().fit_transform(inception_values)

scaled_data_inception.shape



model_95 = PCA(.95)
model_90 = PCA(.90)

model_95.fit(scaled_data_inception)
model_90.fit(scaled_data_inception)


print("95% explained PCA : ",len(model_95.explained_variance_))
print("90% explained PCA : ",len(model_90.explained_variance_))



features = range(model_95.n_components_)


plt.bar(features, model_95.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()




#exception datapath
datapath_exception = "/home/mendel/Documents/data/exception"
#lo9ad the data
frame_ex = load_csv_files(datapath_exception)
#Preprocessing
scaler = StandardScaler()
scaled_data_ex = StandardScaler().fit_transform(frame_ex)
#exception

#(48180, 8985)
scaled_data_ex.shape
#perform PCA
pca_df = pca.fit_transform(scaled_data_ex)
print("90% explained PCA Exception : ",len(pca.explained_variance_))
#90% explained PCA Exception :  302

pca_ratio = [pca for pca in pca.explained_variance_ratio_ if pca >= 0.01]
print(sum(pca_ratio))

labels = ['PC'+ str(x) for x in range(1, len(pca_ratio)+1)]
plt.bar(x=range(1,len(pca_ratio)+1), height=pca_ratio, tick_label =labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('92 Principle Components explaines '
           +str(round(sum(pca_ratio), 2)*100) + "%")
plt.title("PCA: 90% explained PCA Exception :  302  ")
plt.show()




sorted_loading_scores_ex = []
top_feature_indexes_ex = []

#get top 10 important indexes
#alle PCAs die eine Hemschwelle von 0.02 
#überwinden werden als wichtig wahrgenommen
for i in range(10):
  sorted_loading_scores_ex.append(pd.Series(pca.components_[i]).
                                  abs().sort_values(ascending=False))
  
  summe = np.sum(sorted_loading_scores_ex[i] > 0.02)
  
  top_feature_indexes_ex.append(
      sorted_loading_scores_ex[i][:summe].index.values)
  

ex_comic_dict = {
    "pca": pca,
    "top_score_indexes": top_feature_indexes_ex,
    "sorted_score_values": sorted_loading_scores_ex,
    "features": scaled_data_ex
    }


#save pca
#with open('exception_comic_pca.pkl', 'wb') as f:
#  pickle.dump(ex_comic_dict, f)


#load pca
with open('exception_comic_pca.pkl', 'rb') as f:
  dict_ex = pickle.load(f)

pca = dict_ex["pca"]

top_ex_comics = dict_ex["top_score_indexes"]



#top_in_comics = dict_in["top_score_indexes"]



plt.hist(pca.components_[0])
plt.hist(pca.components_[10])

np.sum(pca.components_[0] > 0.02)




pca_in = PCA(0.9)
pca_in.fit_transform

#saved features as csv
datapath_inception = "/home/mendel/Documents/data/inception"
#load data
frame_in = load_csv_files(datapath_inception)
#Preprocessing
scaled_data_in = StandardScaler().fit_transform(frame_in)
scaled_data_in.shape
#perform PCA
pca_df_in = pca_in.fit_transform(scaled_data_in)
print("90% explained PCA Exception : ",len(pca_in.explained_variance_))
#90% explained PCA Exception :  794


pca_ratio_in = [pca for pca in pca_in.explained_variance_ratio_ if pca >= 0.001]
print(sum(pca_ratio_in))

labels_in = ['PC'+ str(x) for x in range(1, len(pca_ratio_in)+1)]
plt.bar(x=range(1,len(pca_ratio_in)+1), 
        height=pca_ratio_in, 
        tick_label =labels_in)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('95 Principle Components explaines ' +
           str(round(sum(pca_ratio_in), 2)*100) + "%")
plt.title("PCA: 90% explained PCA Inception :  794  ")
plt.show()

loading_scores_in = pd.Series(pca_in.components_[0])

sorted_loading_scores = loading_scores_in.abs().sort_values(ascending=False)




sorted_loading_scores_in = []
top_feature_indexes_in= []

#get top 10 important indexes
#alle PCAs die eine Hemschwelle von 0.02 
#überwinden werden als wichtig wahrgenommen
for i in range(10):
  sorted_loading_scores_in.append(pd.Series(pca_in.components_[i]).
                                  abs().sort_values(ascending=False))
  
  summe_in = np.sum(sorted_loading_scores_in[i] > 0.02)
  
  top_feature_indexes_in.append(
      sorted_loading_scores_in[i][:summe_in].index.values)



in_comic_dict = {
    "pca": pca_in,
    "top_score_indexes": top_feature_indexes_in,
    "sorted_score_values": sorted_loading_scores_in,
    "features": scaled_data_in
    }


##save pca
#with open('inception_comic_pca.pkl', 'wb') as f:
#  pickle.dump(in_comic_dict, f)

#load pca
with open('inception_comic_pca.pkl', 'rb') as f:
  dict_in = pickle.load(f)

pca_in = dict_in["pca"]






top_comic_ex = dict_ex["top_score_indexes"]
top_comic_in = dict_in["top_score_indexes"]
#xception = pickle.load(open("/home/jochen/Downloads/auto_xception_Complete_Scans_renamed_rescaled_1024_01_851e58.p", "rb"))
#
#inception = pickle.load(open("/home/jochen/Downloads/auto_inception_v3_Complete_Scans_renamed_rescaled_1024_01_857566.p", "rb"))
#
#xception.keys()
#inception.keys()
#
#
#len(xception['features']) # anzahl der Bilder
#len(xception['features'][0])#anzahl der Layer
#len(xception['features'][0][0])#
#len(xception['features'][0][0][0])#anzahl der channel
#
#
#
#xception['features'][0][0].shape[1]
#
##alle Bilder werden hier geladen
#all_list=[]
#all_df= pd.DataFrame()
#
#index = 0
##anzahl der Bilder
#for bild in range(len(xception['features'])):
#  
#  print("Bild nummer: ", bild)
#  #if bild >= 10000: break #debug nur 1000 Bilder werden geladen
#
#  layer_list = []
#  layer_df = pd.DataFrame()
#
#  #anzahl der Layer
#  for layer in range(len(xception['features'][bild])):
#  
#    activations = np.array(
#        np.zeros(shape=(1,
#                        xception['features'][bild][layer].shape[1])))
#    activations[0,:] = xception['features'][bild][layer][0][:]
#    
#    layer_df = pd.concat([layer_df, pd.DataFrame(activations)], axis= 1)
#    
#    
#    #layer_list.append(activations[0])
#  #all_list.append(layer_list)
# 
#  all_df = pd.concat([all_df, layer_df], axis= 0)
#  print(memory_usage(all_df))
#  
#  if memory_usage(all_df) > 50:
#    index = index + 1
#    
#    all_df.to_csv("/home/mendel/Documents/Features_" + str(index)+".csv")
#    all_df= pd.DataFrame()
#  


#all_df.to_csv("/home/mendel/Documents/Features1.csv")

#model.fit(samples)
#PCA(copy=True)
#
#transformed = model.transform(samples)


#convert into numpy matrix

#loading the features:






  