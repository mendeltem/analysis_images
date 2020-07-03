#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:36:48 2020

@author: mendel
"""
import tensorflow as tf



def simple_classifier(input_shape):
  return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(input_shape,)),
#          tf.keras.layers.Dense(512, activation='relu', dtype='float32'),
#          tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(int(input_shape  /10), activation='relu', dtype='float32'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='softmax')
  ])
      