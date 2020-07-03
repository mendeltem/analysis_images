#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:44:26 2019

preprocessing for image with mask label
it includes a generator

@author: mendel
"""
import os
import random
import numpy as np

import filecmp
import shutil
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.transform import resize
import imageio
import tensorflow as tf
from tensorflow import keras

Model = keras.models.Model
preprocess_input = keras.applications.inception_v3.preprocess_input
EarlyStopping = keras.callbacks.EarlyStopping
ModelCheckpoint = keras.callbacks.ModelCheckpoint
load_model = keras.models.load_model
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
SGD = keras.optimizers.SGD

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from modules.library import get_img_paths

seed = 42

np.random.seed(seed=seed)

#If this is changed. Change same list in module m_preprocess.py
output_choices = ['softmax', 'div_by_sum_and_sigmoid', 'div_by_red_sum',
                  'sigmoid']

def image_mask_generator(trainX,
                         trainY,
                         testX,
                         testY,
                         batch_size,
                         modify_truth = None):
  """Create and return training and test generators.

  Arguments:
    trainX, trainY, testX, testY,  batch_size: self-explanatory

    modify_truth: string to select different truth modification to get
    compatible truth to model output of two or distributions.

  Returns:
    training generator, test generator
  """
 # Image data generator distortion options
  data_gen_args = dict(rotation_range=30.,
                       width_shift_range=0.2,
                       height_shift_range=0.2,
                       shear_range=0.1,
                       zoom_range=0.2,
                       horizontal_flip=True,
                       vertical_flip=True,
                       fill_mode='reflect')  #use 'constant'??
  # Train data, provide the same seed and keyword arguments to the fit and flow methods
  X_datagen = ImageDataGenerator(**data_gen_args)
  Y_datagen = ImageDataGenerator(**data_gen_args)

#  X_datagen.fit(trainX, augment=True, seed=seed)
#  Y_datagen.fit(trainY, augment=True, seed=seed)

  X_train_augmented = X_datagen.flow(trainX,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     seed=seed)
  Y_train_augmented = Y_datagen.flow(trainY,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     seed=seed)
  # Test data, no data augmentation, but we create a generator anyway
  X_datagen_val = ImageDataGenerator()
  Y_datagen_val = ImageDataGenerator()
#  X_datagen_val.fit(testX, augment=False, seed=seed)
#  Y_datagen_val.fit(testY, augment=False, seed=seed)
  X_test_augmented = X_datagen_val.flow(testX,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        seed=seed)
  Y_test_augmented = Y_datagen_val.flow(testY,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        seed=seed)

  def softmax(x):
      e_x = np.exp(x - np.max(x))
      return e_x / e_x.sum()

  if modify_truth == 'softmax':
    def combine_generator(gen1, gen2):
      while True:
        image = gen1.next()
        truth = gen2.next()
        truth_mod = softmax(truth)
        yield (image, truth_mod)
  elif modify_truth == 'div_by_sum_and_sigmoid':
    def combine_generator(gen1, gen2):
      while True:
        image = gen1.next()
        truth = gen2.next()
        truth_mod = truth/np.sum(truth)
        yield (image, [truth_mod, truth])
  elif modify_truth == 'div_by_red_sum':
    def combine_generator(gen1, gen2):
      while True:
        image = gen1.next()
        truth = gen2.next()
        truth_mod = truth/np.sum(truth)
        yield (image, truth_mod)
  else: #default case 'sigmoid'
    def combine_generator(gen1, gen2):
      while True:
        image = gen1.next()
        truth = gen2.next()
        yield (image, truth)

  # combine generators into one which yields image and masks
  train_generator = combine_generator(X_train_augmented, Y_train_augmented)
  test_generator = combine_generator(X_test_augmented, Y_test_augmented)
  return train_generator, test_generator


def read_image(path, img_height, img_width):
  """Read and resize image."""
  img_raw = imageio.imread(path)

  if img_raw.dtype != np.uint8:
    raise Exception('Read in image is not uint8!')

  res_img = resize(img_raw,
                   (int(img_height), int(img_width)),
                   mode='constant',
                   preserve_range=True,
                   anti_aliasing=True,
                   anti_aliasing_sigma=None)

  image = np.zeros((int(img_height),int(img_width),3), dtype=np.uint8)
  if len(img_raw.shape) == 3:
      image = res_img[:,:,:3]
  else:
      image[:,:,0] = res_img[:,:]
      image[:,:,1] = res_img[:,:]
      image[:,:,2] = res_img[:,:]

#  if image.dtype != np.uint8:
#    raise Exception('Read in image is not uint8!')

  return image


def read_mask(path, img_height, img_width):
  """Read and resize mask image."""
  mask_raw = imageio.imread(path)

  if mask_raw.dtype != np.uint8:
    raise Exception('Read in image is not uint8!')

  mask = np.expand_dims(resize(mask_raw,
                               (int(img_height), int(img_width)),
                               mode='constant',
                               preserve_range=True,
                               anti_aliasing=True,
                               anti_aliasing_sigma=None),
                        axis=-1)

#  if mask.dtype != np.uint8:
#    raise Exception('Read in image is not uint8!')

  return mask



def Unet_Data_Generator(data_path,
                        target_size,
                        BATCH_SIZE):
  """Data Generator which provides the model with images directly from the 
  directoy.
  
  
  Arguments:
    data_path: (PATH)       Path to the with training and testing images
    target_size:  (turple)  Image Size:  (height, width)
    BATCH_SIZE:  (int)      Batch size


  Returns:
    training generator, test generator, train_length, test_length
    

  """
  
  paths = {
      'image_train' : data_path+'/train'+'/images',
      'mask_train'  : data_path+'/train'+'/masks',
      'image_test'  : data_path+'/test'+'/images',
      'mask_test'   : data_path+'/test'+'/masks'
      }   
    
  data_gen_args = dict(
     rescale=1./255,
     rotation_range=30.,
     width_shift_range=0.2,
     height_shift_range=0.2,
     shear_range=0.1,
     zoom_range=0.2,
     horizontal_flip=True,
     vertical_flip=True,
     fill_mode='reflect')  #use 'constant'??

  test_gen_args = dict(
     rescale=1./255)
  
  def combine_generator(gen1, gen2, batch_size):
    
      while True:
          generated = (gen1.next(), gen2.next())
           
          if generated[0].shape[0] == batch_size and generated[1].shape[0] == batch_size:
             yield(generated)
  #train Generator
  X_datagen = ImageDataGenerator(**data_gen_args)
  Y_datagen = ImageDataGenerator(**data_gen_args)
  
  #test Generator
  X_datagen_val = ImageDataGenerator(**test_gen_args)
  Y_datagen_val = ImageDataGenerator(**test_gen_args)
  
  X_train_augmented = X_datagen.flow_from_directory(
          paths["image_train"],
          target_size=target_size,
          batch_size=BATCH_SIZE,
          seed=seed,
          class_mode=None)
  
  Y_train_augmented = Y_datagen.flow_from_directory(
          paths["mask_train"],
          target_size=target_size,
          batch_size=BATCH_SIZE,
          seed=seed,
          color_mode = "grayscale",
          class_mode=None)
  
  X_test_augmented = X_datagen_val.flow_from_directory(
          paths["image_test"],
          target_size=target_size,
          batch_size=BATCH_SIZE,
          seed=seed,
          class_mode=None)
  
  Y_test_augmented = Y_datagen_val.flow_from_directory(
          paths["mask_test"],
          target_size=target_size,
          batch_size=BATCH_SIZE,
          seed=seed,
          color_mode = "grayscale",
          class_mode=None)
  
  train_generator = combine_generator(X_train_augmented, 
                                      Y_train_augmented, 
                                      batch_size =  BATCH_SIZE)
  
  test_generator = combine_generator(X_test_augmented, 
                                     Y_test_augmented, 
                                     batch_size =  BATCH_SIZE)
  
  
  train_length  = X_train_augmented.__len__()
  test_length  = X_test_augmented.__len__()
  
  
  return train_generator, test_generator, train_length, test_length




def load_images_and_masks(main_dir, height, width, br=None):
  """Load images and masks into numpy arrays.

  Arguments:
    main_dir: directory containing images and masks in the subdirectories
    images and masks

    height: target image height for resizing

    width: target image width for resizing

    br: number of images and masks to load

  Returns:
    Images and masks as a tuple of two numpy arrays.
  """
  all_image_paths = get_img_paths(main_dir+ "/images/" )
  all_mask_paths = get_img_paths(main_dir+ "/masks/" )

  image_ids = [os.path.basename(i) for i in all_image_paths]
  mask_ids = [os.path.basename(i) for i in all_mask_paths]

  if br:
    image_ids = image_ids[:br]
    mask_ids = mask_ids[:br]

  image_list = []
  mask_list = []
  for n, (image_id, mask_id) in tqdm(enumerate(zip(image_ids, mask_ids)),
                                     total=len(image_ids)):
    try:
      image = read_image(main_dir+'/images/'+image_id, height, width)
      mask = read_mask(main_dir+'/masks/'+mask_id, height, width)
    except Exception as e:
      print('Skipped image and mask:', n, image_id, mask_id, e)
      continue
    image_list.append(np.float16(image/255))
    mask_list.append(np.float16(mask/255))

  X = np.stack(image_list)
  Y = np.stack(mask_list)
  return X, Y

#
#def rotate_allimages(path, degree):
#  """Rotate every images in the given directory, with the given degree
#
#  input: directory path, degree to rotate
#  """
#  image_paths = get_img_paths(path)
#
#  # iterate through the names of contents of the folder
#  for image_path in image_paths:
#    # create the full input path and read the file
#    image_to_rotate = ndimage.imread(image_path)
#    # rotate the image
#    rotated = ndimage.rotate(image_to_rotate, degree)
#    # create full output path, 'example.jpg'
#    # becomes 'rotate_example.jpg', save the file to disk
#    #fullpath = os.path.join(outPath, 'rotated_'+image_id)
#    misc.imsave(image_path, rotated)
#
#

def increment_filename(filename, marker="-"):
    """Appends a counter to a filename, or increments an existing counter."""
    basename, fileext = os.path.splitext(filename)

    # If there isn't a counter already, then append one
    if marker not in basename:
        components = [basename, 1, fileext]

    # If it looks like there might be a counter, then try to coerce it to an
    # integer and increment it. If that fails, then just append a new counter.
    else:
        base, counter = basename.rsplit(marker, 1)
        try:
            new_counter = int(counter) + 1
            components = [base, new_counter, fileext]
        except ValueError:
            components = [base, 1, fileext]

    # Drop in the marker before the counter
    components.insert(1, marker)

    new_filename = "%s%s%d%s" % tuple(components)
    return new_filename

def copyfile(src, dst):
    """Copies a file from path src to path dst.

    If a file already exists at dst, it will not be overwritten, but:

     * If it is the same as the source file, do nothing
     * If it is different to the source file, pick a new name for the copy that
       is distinct and unused, then copy the file there.

    Returns the path to the copy.
    """
    if not os.path.exists(src):
        raise ValueError("Source file does not exist: {}".format(src))

    # Create a folder for dst if one does not already exist
    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))

    # Keep trying to copy the file until it works
    while True:

        # If there is no file of the same name at the destination path, copy
        # to the destination
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
            return dst

        # If the namesake is the same as the source file, then we don't need to
        # do anything else
        if filecmp.cmp(src, dst):
            return dst

        # There is a namesake which is different to the source file, so pick a
        # new destination path
        dst = increment_filename(dst)

    return dst



def asStride(arr,sub_shape,stride):
    '''Get a strided sub-matrices view of an ndarray.
    See also skimage.util.shape.view_as_windows()
    '''
    s0,s1=arr.strides[:2]
    m1,n1=arr.shape[:2]
    m2,n2=sub_shape
    view_shape=(1+(m1-m2)//stride[0],1+(n1-n2)//stride[1],m2,n2)+arr.shape[2:]
    strides=(stride[0]*s0,stride[1]*s1,s0,s1)+arr.strides[2:]
    subs=np.lib.stride_tricks.as_strided(arr,view_shape,strides=strides)
    return subs

def poolingOverlap(mat,ksize,stride=None,method='mean',pad=False):
    '''Overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <stride>: tuple of 2 or None, stride of pooling window.
              If None, same as <ksize> (non-overlapping pooling).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           (n-f)//s+1, n being <mat> size, f being kernel size, s stride.
           if pad, output has size ceil(n/s).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize
    if stride is None:
        stride=(ky,kx)
    sy,sx=stride

    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,sy)
        nx=_ceil(n,sx)
        size=((ny-1)*sy+ky, (nx-1)*sx+kx) + mat.shape[2:]
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
    else:
        mat_pad=mat[:(m-ky)//sy*sy+ky, :(n-kx)//sx*sx+kx, ...]

    view=asStride(mat_pad,ksize,stride)

    if method=='max':
        result=np.nanmax(view,axis=(2,3))
    else:
        result=np.nanmean(view,axis=(2,3))

    return result
  
  
def plot_history(history, save_dir):
  """Ploting the history of the trained model and save the plot in the given 
  directory
  
  Arguments:
   hitory     :  (dictionary from the  keras model)
   save_dir   :  directory where the history image  
 
  """ 
  
  for i, k in enumerate(history.keys()): 
 
    plt.title('Training and Test Normalized Loss')
    plt.plot(history[k]  / np.max(history[k] ), label = k+ ": " + str(round(history[k][-1] , 2)))
    plt.legend()

      
  plt.savefig((save_dir+"history.jpg"))    
  
  
def create_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  
        
        
def create_images(generator , model , save_dir, batch_size, step_size):
  
  create_dir(save_dir)

  name_index = 0
  
  for train_step in range(step_size):
    
    training_images  = next(generator.__iter__())
    
    for i in range(batch_size):
    
      prediction = model.predict(training_images[0])
  
      
      image = np.concatenate(
          (
            np.expand_dims(training_images[0][i][:,:,0], axis = 3),
            training_images[1][i],
            prediction[i]
          ), 
                                 
            axis = 1
      )
      plt.imshow(np.squeeze(image))
  #    plt.savefig(output_dir+'/test/dfs'+str(name_index)+'_out.jpg') 
  #    
  ##     
      image = resize(image,
                     (image.shape[0], image.shape[1]))    
      
      image = image * 255
      image = image.astype(np.uint8)
  
      imageio.imwrite(
          save_dir+str(name_index)+'_out.png', 
          image
          
      ) 
      name_index +=1        