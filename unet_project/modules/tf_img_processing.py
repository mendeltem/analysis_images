#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 17:06:55 2018

@author: DD
"""
import tensorflow as tf
import tensorflow.contrib as tfcontrib

def flip_img(tr_img,
             label_img,
             horizontal_flip = False,
             vertical_flip = False,
             hf_prob = 0.5,
             vf_prob = 0.2):
    """vertical or horizontal flip.

    Arguments:
        horizontal_flip: bool for applying horizontal flip

        vertical_flip: bool for applying vertical flip

        tr_img: image tensor

        label_img: image tensor

        hf_prob: horizontal flip probability

        vf_prob: vertical flip probability

    Returns:
      tuple with training image tensor and label image tensor
    """
    if horizontal_flip:
        flip_prob = tf.random_uniform([], 0.0, 1.0)
        tr_img, label_img = tf.cond(
                                tf.less(flip_prob, hf_prob),
                                lambda: (tf.image.flip_left_right(tr_img),
                                         tf.image.flip_left_right(label_img)),
                                lambda: (tr_img, label_img))
    if vertical_flip:
      flip_prob = tf.random_uniform([], 0.0, 1.0)
      tr_img, label_img = tf.cond(
                                tf.less(flip_prob, vf_prob),
                                lambda: (tf.image.flip_up_down(tr_img),
                                         tf.image.flip_up_down(label_img)),
                                lambda: (tr_img, label_img))

    return tr_img, label_img


def process_pathname(image_filename):
  """

  Arguments:
    image_filename: A Tensor of type string.

  Returns:
    the read in image tensor
  """
  img_str = tf.read_file(image_filename)
  img = tf.image.decode_jpeg(img_str, channels=3)
  return img


def augment(img,
            label_img,
            img_shape,
            resize=None,
            normalize_values=True,
            hue_delta=0,
            horizontal_flip=False,
            vertical_flip = False,
            width_shift_range=0,
            height_shift_range=0):
  if resize is not None:
    # Resize both images
    label_img = tf.image.resize_images(label_img, resize)
    img = tf.image.resize_images(img, resize)

  if hue_delta:
    img = tf.image.random_hue(img, hue_delta)

  if not all([dim != None for dim in img_shape]):
    img_shape = tf.to_float(tf.shape(img))

  img, label_img = flip_img(img, label_img, horizontal_flip, vertical_flip)
  img, label_img = shift_img(img,
                             label_img,
                             img_shape,
                             width_shift_range,
                             height_shift_range)
  if normalize_values is True:
    label_img = tf.to_float(label_img)
    img = tf.to_float(img) / 255
#    label_img = tf.to_float(label_img)
#    img = tf.to_float(img) / 127.5 - 1
  return img, label_img

def shift_img(output_img,
              label_img,
              img_shape,
              width_shift_range,
              height_shift_range):
  """This fn will perform the horizontal or vertical shift"""
  if width_shift_range or height_shift_range:
      if width_shift_range:
        width_shift_range = tf.random_uniform(
            [],
            -width_shift_range * img_shape[1],
            width_shift_range * img_shape[1])
      if height_shift_range:
        height_shift_range = tf.random_uniform(
            [],
            -height_shift_range * img_shape[0],
            height_shift_range * img_shape[0])
      # Translate both
      output_img = tfcontrib.image.translate(output_img,
                                             [width_shift_range,
                                              height_shift_range])
      label_img = tfcontrib.image.translate(label_img,
                                             [width_shift_range,
                                              height_shift_range])
  return output_img, label_img