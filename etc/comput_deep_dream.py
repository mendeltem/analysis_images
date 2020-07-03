#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:35:26 2019

@author: mendel
"""
from __future__ import print_function

import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

import keras

from keras.preprocessing.image import load_img, img_to_array


from sklearn.datasets import load_sample_images
import numpy as np
import scipy

from keras.applications import inception_v3
from keras import backend as K

import matplotlib.pyplot as plt

base_image_path = "comic_picture/AmericanBornChi__00009_small.png"
result_prefix   = "dream_"
import os


##load pca
#with open('exception_comic_pca.pkl', 'rb') as f:
#  dict_ex = pickle.load(f)


#load pca
with open('inception_comic_pca.pkl', 'rb') as f:
  dict_in = pickle.load(f)
  
#
##top_ex_comics       = dict_ex["top_score_indexes"]
top_in_comics       = dict_in["top_score_indexes"]



#load the model
model = inception_v3.InceptionV3(weights='imagenet',
                                 include_top=False)

#get the shape list 
layer_list = [(layer.output_shape[3]) for layer in model.layers 
                   if "mixed" in layer.name and "_" not in layer.name]


mixed = []
layer = []

#get the featurenumber and mixlayers from pure channelnumber
#Example 700 - 256 - 288 - 
for i,channel in enumerate(top_in_comics[0]):
  for z,shape in enumerate(layer_list):
    if channel >= shape:
      channel = channel - shape
    else:
      mixed.append(channel)
      layer.append(["mixed"+str(z)])
      break



#get only high level layers    
for i,channel in enumerate(top_in_comics[0]):
  for z,shape in enumerate(layer_list):
    if channel >= shape:
      channel = channel - shape
    else:
      if z > 5:
        mixed.append(channel)
        layer.append(["mixed"+str(z)])
      break
    
    
    
def dream(base_image_path, result_prefix, feature_number, mixed_layers):


    # These are the names of the layers
    # for which we try to maximize activation,
    # as well as their weight in the final loss
    # we try to maximize.
    # You can tweak these setting to obtain new visual effects.
    settings = {
        'features': {
    #        'mixed0': 1,
    #         'mixed3': 0.5,
    #        'mixed4': 2.,
    #        'mixed5': 1.5,
        },
    }
    

    for ml in mixed_layers:
        settings['features'][ml] = 1

    def preprocess_image(image_path):
        # Util function to open, resize and format pictures
        # into appropriate tensors.
        img = plt.imread(image_path)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = inception_v3.preprocess_input(img)
        return img


    def deprocess_image(x):
        # Util function to convert a tensor into a valid image.
        if K.image_data_format() == 'channels_first':
            x = x.reshape((3, x.shape[2], x.shape[3]))
            x = x.transpose((1, 2, 0))
        else:
            x = x.reshape((x.shape[1], x.shape[2], 3))
        x /= 2.
        x += 0.5
        x *= 255.
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    K.set_learning_phase(0)

    # Build the InceptionV3 network with our placeholder.
    # The model will be loaded with pre-trained ImageNet weights.
    model = inception_v3.InceptionV3(weights='imagenet',
                                     include_top=False)
    dream = model.input
    print('Model loaded.')

    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    # Define the loss.
    loss = K.variable(0.)
    for layer_name in settings['features']:
        # Add the L2 norm of the features of a layer to the loss.
        assert layer_name in layer_dict.keys(), 'Layer ' + layer_name + ' not found in model.'
        coeff = settings['features'][layer_name]
        x = layer_dict[layer_name].output
        # We avoid border artifacts by only involving non-border pixels in the loss.
        scaling = K.prod(K.cast(K.shape(x), 'float32'))
        if K.image_data_format() == 'channels_first':
            loss += coeff * K.sum(K.square(x[:, feature_number, 2: -2, 2: -2])) / scaling
        else:
            loss += coeff * K.sum(K.square(x[:, 2: -2, 2: -2, feature_number])) / scaling

    # Compute the gradients of the dream wrt the loss.
    grads = K.gradients(loss, dream)[0]
    # Normalize gradients.
    grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())

    # Set up function to retrieve the value
    # of the loss and gradients given an input image.
    outputs = [loss, grads]
    fetch_loss_and_grads = K.function([dream], outputs)


    def eval_loss_and_grads(x):
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1]
        return loss_value, grad_values


    def resize_img(img, size):
        img = np.copy(img)
        if K.image_data_format() == 'channels_first':
            factors = (1, 1,
                       float(size[0]) / img.shape[2],
                       float(size[1]) / img.shape[3])
        else:
            factors = (1,
                       float(size[0]) / img.shape[1],
                       float(size[1]) / img.shape[2],
                       1)
        return scipy.ndimage.zoom(img, factors, order=1)


    def gradient_ascent(x, iterations, step, max_loss=None):
        for i in range(iterations):
            loss_value, grad_values = eval_loss_and_grads(x)
            if max_loss is not None and loss_value > max_loss:
                break
            print('..Loss value at', i, ':', loss_value)
            x += step * grad_values
        return x


    def save_img(img, fname):
        pil_img = deprocess_image(np.copy(img))
        scipy.misc.imsave(fname, pil_img)


    """Process:
    - Load the original image.
    - Define a number of processing scales (i.e. image shapes),
        from smallest to largest.
    - Resize the original image to the smallest scale.
    - For every scale, starting with the smallest (i.e. current one):
        - Run gradient ascent
        - Upscale image to the next scale
        - Reinject the detail that was lost at upscaling time
    - Stop when we are back to the original size.
    To obtain the detail lost during upscaling, we simply
    take the original image, shrink it down, upscale it,
    and compare the result to the (resized) original image.
    """


    # Playing with these hyperparameters will also allow you to achieve new effects
    step = 0.01  # Gradient ascent step size
    num_octave = 3  # Number of scales at which to run gradient ascent
    octave_scale = 1.4  # Size ratio between scales
    iterations = 20  # Number of ascent steps per scale
    max_loss = 10.
    
    
     #Original:
#    # Playing with these hyperparameters will also allow you to achieve new effects
#    step = 0.01  # Gradient ascent step size
#    num_octave = 3  # Number of scales at which to run gradient ascent
#    octave_scale = 1.4  # Size ratio between scales
#    iterations = 20  # Number of ascent steps per scale
#    max_loss = 10.
     
    img = preprocess_image(base_image_path)
    if K.image_data_format() == 'channels_first':
        original_shape = img.shape[2:]
    else:
        original_shape = img.shape[1:3]
    successive_shapes = [original_shape]
    for i in range(1, num_octave):
        shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
        successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1]
    original_img = np.copy(img)
    shrunk_original_img = resize_img(img, successive_shapes[0])

    for shape in successive_shapes:
        print('Processing image shape', shape)
        img = resize_img(img, shape)
        img = gradient_ascent(img,
                              iterations=iterations,
                              step=step,
                              max_loss=max_loss)
        upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
        same_size_original = resize_img(original_img, shape)
        lost_detail = same_size_original - upscaled_shrunk_original_img

        img += lost_detail
        shrunk_original_img = resize_img(original_img, shape)

    save_img(img, fname=result_prefix +
             "fn_"+
             str(feature_number)+
             "_"+mixed_layers[0] +
             "_"+os.path.basename(base_image_path))
    
    
    
#
#    
#    
#dream(base_image_path, result_prefix, 433, ["mixed3"])



for i, mix in enumerate(mixed): 
  print(layer[i])
  print(mix)
  print(top_in_comics[0][i])
  print("next")
  dream(base_image_path, result_prefix, mix,  layer[i])