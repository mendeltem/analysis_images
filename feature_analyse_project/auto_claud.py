#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:16:21 2020

@author: mendel
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping

## pandas dataframe
# all_data = pd.read_csv("all_data.csv")
# all_data = pd.read_csv("/home/fisibubele/Dokumente/Studium/Uni_Potsdam/SHK/Hybride_Narrativität/Featureanalyse/Dimensionsreduktion/all_data.csv", nrows=1000)
# all_data.to_pickle("all_data.pkl")

with open( '/home/claud/Dokumente/all_data_pd.pkl', 'rb') as df:
   all_data = pickle.load(df)

# with open('all_data_norm.pkl', 'rb') as df:
#    all_data = pickle.load(df)

# normalisation
# all_data_norm = all_data
# all_data_norm.iloc[:,1:10253] = StandardScaler().fit_transform(all_data_norm.iloc[:,1:10253])
# all_data_norm.to_pickle("all_data_norm.pkl")


## shape of data:
# first column: png name
# inception features start from column 206-10253
# normalisation 
# delete first column
all_data_loaded = all_data.iloc[:,206:10254]


all_data_norm = StandardScaler().fit_transform(all_data_loaded) # now data type changed to numpy array

## definition of autoencoder
# size of encoded representations
encoding_dim = 256  # compression factor ~ 40

# input placeholder
input_png = Input(shape=(all_data_norm.shape[1],))
# encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_png) # activation function transfers input signals of neurons
# lossy reconstruction of the input
decoded = Dense(all_data_norm.shape[1], activation='sigmoid')(encoded)

## models
# maps an input to its reconstruction
autoencoder = Model(input_png, decoded)
# maps an input to its encoded representation
encoder = Model(input_png, encoded)

# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy') # only if values in [0,1]
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True, mode='min')
# early_stopping = EarlyStopping(monitor="val_acc", patience=3, restore_best_weights=True, mode='max')
# autoencoder.fit(all_data_norm, all_data_norm, epochs = 50, batch_size = 256, shuffle = True, validation_data = (all_data_norm, all_data_norm), callbacks=[early_stopping])
history = autoencoder.fit(all_data_norm, all_data_norm, batch_size=256, shuffle=True, validation_split=0.3,epochs = 100, callbacks=[early_stopping]) # train until no improvement for three epochs

## saving and loading model 
autoencoder.save('autoencoder_256_adadelta_mse.h5')
encoder.save('encoder_256_adadelta_mse.h5')

# import keras
# autoencoder = keras.models.load_model('autoencoder.h5')
# encoder = keras.models.load_model('encoder.h5')

# ## saving in json format
# json_model = autoencoder.to_json()
# json_file = open('autoencoder_json.json', 'w')
# json_file.write(json_model)

# ## loading model architecture from json file
# from keras.models import model_from_json
# json_file = open('autoencoder_json.json', 'r')
# json_model = model_from_json(json_file.read())

## generate encoded data set
# all_data_encoded = encoder.predict(all_data_norm)

## tsne
# from sklearn.manifold import TSNE