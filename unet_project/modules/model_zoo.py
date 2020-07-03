#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:19:36 2018

@author: DD

contains models, buildings blocks for models, loss functions and metrics
using tensorflow
"""
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.models import Model


def encoder_block(input_tensor,
                  num_filters,
                  pool = True,
                  bn_after_act = True,
                  reg = False):
  """Encoder block for the encoder decoder model.

  Arguments:
    input_tensor: tensor

    num_filters: number of filters / channels for output

    bn_after_act: applies batch normalization after activation. otherwise
    it is applied the other way around.

    reg: if true, uses l2 kernel regularization in encoder

  Returns:
    tensor
  """
  if pool is True:
      encoder = layers.MaxPooling2D((2, 2),
                                    strides=(2, 2),
                                    padding='same')(input_tensor)
  else:
      encoder = input_tensor
  encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
  encoder = activation_with_bn(encoder, bn_last = bn_after_act)

  if reg:
#    encoder = layers.Conv2D(num_filters, (3, 3), padding='same',
#    activity_regularizer=regularizers.l1(0.001))(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(0.001))(encoder)
  else:
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
  encoder = activation_with_bn(encoder, bn_last = bn_after_act)
  return encoder


def decoder_block(tensor_decode,
                  tensor_concat,
                  num_filters,
                  bn_after_act = True,
                  reg = False):
  """Decoder block for the encoder decoder model.

  Arguments:
    decode_tensor: tensor as input for the transpose convolution

    concat_tensor: tensor used to concatenate with the deconvoluted
    tensor

    num_filters: number of filters / channels for output

    bn_after_act: applies batch normalization after activation. otherwise
    it is applied the other way around.

    reg: if evaluate as True, uses l2 kernel regularization in decoder with
    parameter reg as a float. Often used value for regularization is 0.001.

  Returns:
    tensor
  """
  x = layers.Conv2DTranspose(num_filters,
                             (2, 2),
                             strides=(2, 2),
                             padding='same')(tensor_decode)
  if not tensor_concat is None:
    x = layers.concatenate([tensor_concat, x], axis=-1)
  x = activation_with_bn(x, bn_last = bn_after_act)
  x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
  x = activation_with_bn(x, bn_last = bn_after_act)
  if reg:
    x = layers.Conv2D(num_filters, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(reg))(x)
  else:
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
  x = activation_with_bn(x, bn_last = bn_after_act)
  return x


def activation_with_bn(input_tensor,
                       activation_name = 'relu',
                       use_batch_normalization = True,
                       bn_last = True):
  if not bn_last is True:
    x = layers.BatchNormalization()(input_tensor)
  else:
    x = input_tensor
  x = layers.Activation(activation_name)(x)
  if bn_last is True:
    x = layers.BatchNormalization()(x)
  return x


def encoder_decoder_model(img_shape, reg_enc = False, reg_dec = False):
  """Simple encoding decoding model (UNet). Encodes by factor 2**4 and
  decodes back.

  Arguments:
    img_shape: image shape as (height, width, channels)

    reg_enc: regularize encoding

    reg_dec: regularize decoding

  Returns:
    a keras model
  """
  inputs = layers.Input(shape=img_shape)
  # (256,256,3)
  encoder0 = encoder_block(inputs, 32, pool=False, reg=reg_enc)
  # (256,256,32)
  encoder1 = encoder_block(encoder0, 64, reg=reg_enc)
  # (128,128,64)
  encoder2 = encoder_block(encoder1, 128, reg=reg_enc)
  # (64,64,128)
  encoder3 = encoder_block(encoder2, 256, reg=reg_enc)
  # (32,32,256)
  encoder4 = encoder_block(encoder3, 512, reg=reg_enc)
  # (16,16,512)
  center = encoder_block(encoder4, 1024, reg=reg_enc)
  # (8,8,1024)
  decoder4 = decoder_block(center, encoder4, 512, reg=reg_dec)
  # (16,16,512)
  decoder3 = decoder_block(decoder4, encoder3, 256, reg=reg_dec)
  # (32,32,256)
  decoder2 = decoder_block(decoder3, encoder2, 128, reg=reg_dec)
  # (64,64,128)
  decoder1 = decoder_block(decoder2, encoder1, 64, reg=reg_dec)
  # (128,128,64)
  decoder0 = decoder_block(decoder1, encoder0, 32, reg=reg_dec)
  # (256,256,32)
  outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)
  # (256,256,1)
  model = models.Model(inputs=[inputs], outputs=[outputs])
  return model


def vgg16_enc_dec_model(img_shape, reg_dec = False, freeze_vgg=True):
  """Simple vgg16 based encoding and learned decoding model (UNet).
  Encodes by factor 2**5 and decodes back.

  Arguments:
    img_shape: image shape as (height, width, channels)

    reg_dec: float for regularization in the decoding parts of the network.
    If evaluated as False, no regularization. Otherwise regularization with
    this parameter. Often used is 0.001.

  Returns:
    a tf keras model
  """
  inputs = layers.Input(shape=img_shape)
  # (256,256,3)
  vgg16_encoding = vgg16_enc_model(freeze=freeze_vgg)
  encoder0, encoder1, encoder2, encoder3, encoder4 = vgg16_encoding(inputs)
  # (128,128,64), (64,64,128), (32,32,256), (16,16,512), (8,8,512)
  decoder3 = decoder_block(encoder4, encoder3, 256, reg=reg_dec)
  # (16,16,256)
  decoder2 = decoder_block(decoder3, encoder2, 128, reg=reg_dec)
  # (32,32,128)
  decoder1 = decoder_block(decoder2, encoder1, 64, reg=reg_dec)
  # (64,64,64)
  decoder0 = decoder_block(decoder1, encoder0, 32, reg=reg_dec)
  # (128,128,32)
  final_up = decoder_block(decoder0, None, 32, reg=reg_dec)
  # (256,256,32)
  outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(final_up)
  # (256,256,1)
  model = Model(inputs=[inputs], outputs=[outputs])
  return model



def dice_coeff(y_true, y_pred):
  """Dice coefficient for dice loss function.

  Arguments:
    y_true: label tensor of dtype float.

    y_pred: predicted tensor of dtype float.

  Returns:
    a scalar tensor of dtype float
  """
  smooth = 1.
  # Flatten
  y_true_f = tf.reshape(y_true, [-1])
  y_pred_f = tf.reshape(y_pred, [-1])
  intersection = tf.reduce_sum(y_true_f * y_pred_f)
  score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) +
           tf.reduce_sum(y_pred_f) + smooth)
  return score


def bce_loss(y_true, y_pred):
  """Binary cross entropy loss function.

  Arguments:
    y_true: label tensor of dtype float.

    y_pred: predicted tensor of dtype float.

  Returns:
    a scalar tensor of dtype float
  """
  loss = losses.binary_crossentropy(y_true, y_pred)
  return loss


def dice_loss(y_true, y_pred):
  """Dice loss function.

  Arguments:
    y_true: label tensor of dtype float.

    y_pred: predicted tensor of dtype float.

  Returns:
    a scalar tensor of dtype float
  """
  loss = 1 - dice_coeff(y_true, y_pred)
  return loss


def bce_dice_loss(y_true, y_pred):
  """loss function consisting of binary cross entropy plus dice loss

  Arguments:
    y_true: label tensor of dtype float.

    y_pred: predicted tensor of dtype float.

  Returns:
    a scalar tensor of dtype float
  """
  loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
  return loss


def precision(y_true, y_pred):
  """compute precision for prediction vs truth

  Arguments:
    y_true: a tensor of 0 and 1 providing the ground truth

    y_pred: a tensor with the same shape as y_true with floating
    point values between 0 and 1

  Returns:
    the precision as a floating point value
  """
  y_pred_bool = tf.math.greater(y_pred, 0.5)
  true_positive = tf.boolean_mask(y_true, y_pred_bool)
  y_true_inverted = tf.abs(tf.add(y_true, -1))
  false_positive = tf.boolean_mask(y_true_inverted, y_pred_bool)
  n_tp = tf.reduce_sum(true_positive)
  n_fp = tf.reduce_sum(false_positive)
  return n_tp/(n_tp + n_fp)


def recall(y_true, y_pred):
  """compute recall for prediction vs truth

  Arguments:
    y_true: a tensor of 0 and 1 providing the ground truth

    y_pred: a tensor with the same shape as y_true with floating
    point values between 0 and 1

  Returns:
    the recall as a floating point value
  """
  y_pred_bool = tf.math.greater(y_pred, 0.5)
  true_positive = tf.boolean_mask(y_true, y_pred_bool)
  y_pred_bool_neg = tf.logical_not(y_pred_bool)
  false_negative = tf.boolean_mask(y_true, y_pred_bool_neg)
  n_tp = tf.reduce_sum(true_positive)
  n_fn = tf.reduce_sum(false_negative)
  return n_tp/(n_tp + n_fn)


def incv3_enc_model(wanted_mixed_layers=None):
  """Returns a model that has all mixed layers of Inception V3 as output.
  """
  if not wanted_mixed_layers:
    wanted_mixed_layers = ['mixed'+str(i) for i in range(11)]
  from tensorflow.python.keras.applications import InceptionV3
  inc_v3 = InceptionV3(include_top=False, weights='imagenet')
  mixed_layer_names = [layer.name for layer in inc_v3.layers if 'mixed'
                       in layer.name and '_' not in layer.name]
  layers = [l for l in wanted_mixed_layers if l in mixed_layer_names]
  return Model(inputs = inc_v3.input, outputs = [inc_v3.get_layer(ln).output
                                                 for ln in layers] )


def vgg16_enc_model(output_layers = None, freeze=True):
  """Returns a model that has all block pool layers of VGG16 as output.
  """
  if not output_layers:
    output_layers = ['block' + str(i) + '_pool' for i in range(1,6)]
  from tensorflow.python.keras.applications import VGG16
  vgg16 = VGG16(include_top=False, weights='imagenet')
  if freeze:
    for layer in vgg16.layers:
      layer.trainable = False
    
  pool_layer_names = ['block' + str(i) + '_pool' for i in range(1,6)]
  layers = [l for l in output_layers if l in pool_layer_names]
  return Model(inputs=vgg16.input, outputs = [vgg16.get_layer(ln).output
                                                 for ln in layers] )