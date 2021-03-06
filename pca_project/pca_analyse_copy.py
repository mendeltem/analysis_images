#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:52:52 2019

@author: mendel
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from modules.library import get_img_paths,get_feature_paths,memory_usage


xception = pickle.load(open("/home/jochen/Downloads/auto_xception_Complete_Scans_renamed_rescaled_1024_01_851e58.p", "rb"))

inception = pickle.load(open("/home/jochen/Downloads/auto_inception_v3_Complete_Scans_renamed_rescaled_1024_01_857566.p", "rb"))

feature_object = inception

xception.keys()
inception.keys()

len(feature_object['features']) # anzahl der Bilder
len(feature_object['features'][0])#anzahl der Layer
len(feature_object['features'][0][0])#
len(feature_object['features'][0][0][0])#anzahl der channel

feature_object['features'][0][0].shape[1]

#alle Bilder werden hier geladen
all_list=[]
all_df= pd.DataFrame()

index = 0
#anzahl der Bilder
for bild in range(len(feature_object['features'])):
  
  print("Bild nummer: ", bild)
  #if bild >= 10000: break #debug nur 1000 Bilder werden geladen

  layer_list = []
  layer_df = pd.DataFrame()

  #anzahl der Layer
  for layer in range(len(feature_object['features'][bild])):
  
    activations = np.array(
        np.zeros(shape=(1,
                        feature_object['features'][bild][layer].shape[1])))
    activations[0,:] = feature_object['features'][bild][layer][0][:]
    
    layer_df = pd.concat([layer_df, pd.DataFrame(activations)], axis= 1)
    
  all_df = pd.concat([all_df, layer_df], axis= 0)
  print(memory_usage(all_df))
  
  if memory_usage(all_df) > 50:
    index = index + 1
    
    all_df.to_csv("/home/mendel/Documents/inception_Features_" + str(index)+".csv")
    all_df= pd.DataFrame()


#convert into numpy matrix
numpy_matrix = all_df.as_matrix()
scaler = StandardScaler()

#scaler
x = StandardScaler().fit_transform(numpy_matrix)

#95%exxplained
model = PCA(.95)
model.fit(x)
pca_features = model.transform(x)
model.explained_variance_ratio_
print("95% explained PCA : ",len(model.explained_variance_))


#90 % explained
model_90 = PCA(.90)
model_90.fit(x)
pca_features = model_90.transform(x)
model_90.explained_variance_ratio_
len(model_90.explained_variance_)
print("90% explained PCA : ",len(model_90.explained_variance_))
# Plot the explained variances
features = range(model_90.n_components_)

plt.bar(features, model_90.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()









  
