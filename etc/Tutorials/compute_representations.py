#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 15:17:26 2018

@author: DD

compute and save intermediate representations of images from the
inception_v3 net.
"""
import pickle
import keras
import os

from keras.applications.inception_v3 import preprocess_input
from modules.generators import ImageSequence

DEFAULT_SAVE_PATH_MODEL = 'model_inception_v3_mixedlayeroutputs_auto.h5'
DEFAULT_IMAGE_DIRECTORY = 'images/color/session1_memory/high-pass'
DEFAULT_SAVE_DATA_PATH = 'reps_incv3_mixedlayers_auto.p'

SAVE_PATH_MODEL = ''
IMAGE_DIRECTORY = ''
SAVE_DATA_PATH = ''
#batch size can be higher than one if images have same size.
BATCH_SIZE = 1

def get_model(load_path = None, auto_save = True):
    """Loads or instantiates a model based on inception_v3 returning the
    outputs of all 11 mixed layers with indices from 0 to 10.

    Arguments:
        load_path: path of the already instantiated saved model.

        auto_save: if True, saves the model in the default save path, if model
        was not loaded.

    Returns:
        a keras model with all full mixed layers of inception_V3 as output
    """
    if load_path is None:
        if SAVE_PATH_MODEL == '':
            load_path = DEFAULT_SAVE_PATH_MODEL
        else:
            load_path = SAVE_PATH_MODEL
    try:
        model = keras.models.load_model(load_path)

    except OSError:
        inc_v3 = keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')

        def get_mixed_layer_names():
            layer_names = []
            for layer in inc_v3.layers:
                if 'mixed' in layer.name:
                    layer_names.append(layer.name)
            return layer_names

        mixed_layer_names = get_mixed_layer_names()

        main_mixed_layer_names = [ln for ln in mixed_layer_names if '_' not in ln]

        x = inc_v3.input
        outs = []
        for ln in main_mixed_layer_names:
            outs.append(inc_v3.get_layer(ln).output)
        model = keras.Model(inputs=x, outputs=outs)
        if auto_save:
            model.save(DEFAULT_SAVE_PATH_MODEL, include_optimizer=False)
    return model

def get_img_paths(start_dir, extensions = ['png']):
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

model = get_model()

if IMAGE_DIRECTORY == '':
    img_paths = get_img_paths(DEFAULT_IMAGE_DIRECTORY)
else:
    img_paths = get_img_paths(IMAGE_DIRECTORY)

img_sequ = ImageSequence(paths = img_paths,
                         labels = None,
                         batch_size = BATCH_SIZE,
                         preprocessing=preprocess_input,
                         augmentation=[])

predictions = model.predict_generator(img_sequ,
                                      verbose=1,
                                      steps=None,
                                      use_multiprocessing=True)

total = {'paths':img_paths,
         'mixed_layer_represenations':predictions}

#python has a problem with pickle.dump and very large files
def write(data, file_path):
    """Writes data (using pickle) to a file in smaller byte chunks.

    Arguments:
        data: the data to be pickled

        file_path: the file to be written to (or created)
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(total)
    with open(file_path, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])

if SAVE_DATA_PATH == '':
    write(total, DEFAULT_SAVE_DATA_PATH)
else:
    write(total, SAVE_DATA_PATH)