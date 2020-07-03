#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:00:01 2019

@author: DD

contains models, loss functions, model build blocks, metrics and utility
functions for these

see model dictionary for easy model selection
"""
import numpy as np
import tensorflow as tf
keras = tf.keras
layers = keras.layers
losses = keras.losses
models = keras.models
regularizers = keras.regularizers
Model = keras.models.Model


def spatial_dims(tensor):
  """Get the spatial dimensions as a integer tuple for a tensor
  in channel last mode with a batch dimension (B, H, W, C).
  """
  return (int(tensor.shape[1]), int(tensor.shape[2]))


def padded_transposed_conv(tensor,
                           target_dim,
                           kernel_regularizer=None,
                           bias_regularizer=None):
  """Upsample the tensor to the target dimension.
  Uses transposed convolution. Pads the incoming tensor and the result of
  the transposed convolution to fit the target dimensions.

  Arguments:
    tensor: the incoming tensor

    target_dim: the target dimensions as a tuple of three integers (Height,
    Width, Channels)

    kernel_regularizer: kernel regularizer for the transposed convolution

    bias_regularizer: bias regularizer for the transposed convolution

  Returns:
    the convoluted and padded result tensor with dimensions as given in
    target_dim
  """
  def params_1dim(in_dim, out_dim):
    """Parameters for padded transposed convolution in one spatial dimension.

    Arguments:
      in_dim: input dimension as an integer

      out_dim: output dimension as an integer

    Returns:
      stride, kernel, pad_in, pad_out
    """
    up_factor, remaining_pixels = np.divmod(out_dim, in_dim)
    pad_in, pad_out = np.divmod(remaining_pixels, up_factor)
    kernel = up_factor
    stride = up_factor
    return stride, kernel, pad_in, pad_out

  def split_padding(padding):
    res, rem = np.divmod(padding, 2)
    return (res, res + rem)

  in_sdims = spatial_dims(tensor)

  out_sdims = (target_dim[0], target_dim[1])
  out_filters = target_dim[2]

  stride_0, kernel_0, pad_in_0, pad_out_0 = \
    params_1dim(in_sdims[0], out_sdims[0])
  pad_in_0 = split_padding(pad_in_0)
  pad_out_0 = split_padding(pad_out_0)

  stride_1, kernel_1, pad_in_1, pad_out_1 = \
    params_1dim(in_sdims[1], out_sdims[1])
  pad_in_1 = split_padding(pad_in_1)
  pad_out_1 = split_padding(pad_out_1)

  x = layers.ZeroPadding2D((pad_in_0, pad_in_1))(tensor)

  x = layers.Conv2DTranspose(filters = out_filters,
                             kernel_size = (kernel_0, kernel_1),
                             strides = (stride_0, stride_1),
                             padding = 'same',
                             kernel_regularizer = kernel_regularizer,
                             bias_regularizer = bias_regularizer)(x)

  x = layers.ZeroPadding2D((pad_out_0, pad_out_1))(x)
  return x


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
    tensor. has to fit the deconvoluted tensor.

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


def dyn_decoder_block(tensor_decode,
                      tensor_concat,
                      num_filters = None,
                      target_dim = None,
                      bn_after_act = True,
                      reg = False):
  """Dynamic decoder block for the encoder decoder model.

  Arguments:
    decode_tensor: tensor as input for the transpose convolution

    concat_tensor: tensor used to concatenate with the deconvoluted
    tensor. defines the transposed convolution dimensions.

    num_filters: number of filters / channels for output

    bn_after_act: applies batch normalization after activation. otherwise
    it is applied the other way around.

    reg: if evaluate as True, uses l2 kernel regularization in decoder with
    parameter reg as a float. Often used value for regularization is 0.001.

  Returns:
    decoded tensor
  """
  if not tensor_concat is None:
    x = padded_transposed_conv(tensor_decode,
                               (int(tensor_concat.shape[1]),
                                int(tensor_concat.shape[2]),
                                int(tensor_concat.shape[3])))
    x = layers.concatenate([tensor_concat, x], axis=-1)
  elif target_dim:
    x = padded_transposed_conv(tensor_decode,
                               target_dim)
  else:
    x = padded_transposed_conv(tensor_decode,
                               (int(tensor_decode.shape[1]),
                                int(tensor_decode.shape[2]),
                                num_filters))

  if not num_filters:
    if target_dim:
      num_filters = target_dim[2]
    else:
      num_filters = int(tensor_concat.shape[3])

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


def dyn_enc_dec_model(img_shape,
                      encoder,
                      reg_dec = False):
  """Simple encoding decoding model.
  Decoding can be any tensors that can be interpreted as a ordered encoding.
  Based on unet.

  Arguments:
    img_shape: image shape as (height, width, channels)

    reg_dec: float for regularization in the decoding parts of the network.
    If evaluated as False, no regularization. Otherwise regularization with
    this parameter. Often used is 0.001.

  Returns:
    a tf keras model
  """
  inputs = layers.Input(shape=img_shape)
  encodings = sorted(encoder(inputs),
                     key = lambda x:(x.shape[1],x.shape[2],x.shape[3]),
                     reverse = False)
  for n,e in enumerate(encodings):
    if n + 1 < len(encodings) and n == 0:
      decoded = dyn_decoder_block(e, encodings[n+1])
    elif n + 1 < len(encodings):
      decoded = dyn_decoder_block(decoded, encodings[n+1])
    else:
      decoded = dyn_decoder_block(decoded, None, target_dim = img_shape)

  outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoded)
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


def incv3_enc_model(wanted_layers=[0,1,4,9,14]):
  """Returns a model that has the mixed layers and two earliers of
  Inception V3 as output.

  Arguments:
    wanted_layers: Either a list of integers to select from the standard
    selection of output layers or none.

  Returns:
    model with possibly multiple outputs
  """
  from keras.applications import InceptionV3
  inc_v3 = InceptionV3(include_top=False, weights='imagenet')

  selection = [layer for layer in inc_v3.layers if 'mixed' in layer.name
               or layer.name == 'activation_2' or layer.name == 'activation_4']

  if not wanted_layers:
    wanted_layers = selection
  else:
    if isinstance(wanted_layers[0], int):
      wanted_layers = [l.name for n,l in enumerate(selection)
                              if n in wanted_layers]

  return Model(inputs = inc_v3.input, outputs = [inc_v3.get_layer(ln).output
                                                 for ln in wanted_layers] )


def xception_enc_model(wanted_layers=[0, 5, 6, 24, 27]):
  """Xception based encoder model with preselected output layers."""
  from keras.applications import Xception
  xception = Xception(include_top = False, weights='imagenet')

  sepconv2_layers = [l for l in xception.layers if 'block' in l.name and
                     'conv2_bn' in l.name or 'conv1_bn' in l.name]

  if not wanted_layers:
    wanted_layers = sepconv2_layers
  else:
    if isinstance(wanted_layers[0], int):
      wanted_layers = [l.name for n,l in enumerate(sepconv2_layers)
                        if n in wanted_layers]

  return Model(inputs = xception.input,
               outputs = [xception.get_layer(lname)
                          for lname in wanted_layers])


def resnet_50_enc_model(wanted_layers=[0, 2, 7, 13, 16]):
  """Resnet50 based encoder model with preselected output layers."""
  from keras.applications.resnet50 import ResNet50
  resnet50 = ResNet50(include_top=False, weights='imagenet')

  add_layers = [l for l in resnet50.layers if 'add' in l.name or
                l.name == 'bn_conv1']

  if not wanted_layers:
    wanted_layers = add_layers
  else:
    if isinstance(wanted_layers[0], int):
      wanted_layers = [l.name for n,l in enumerate(add_layers)
                        if n in wanted_layers]

  return Model(inputs = resnet50.input,
               outputs = [resnet50.get_layer(lname)
                          for lname in wanted_layers])


def vgg16_enc_model(output_layers=None, freeze=True):
  """Returns a model that has all block pool layers of VGG16 as output.
  """
  if not output_layers:
    output_layers = ['block' + str(i) + '_pool' for i in range(1,6)]
  from keras.applications import VGG16
  vgg16 = VGG16(include_top=False, weights='imagenet')
  if freeze:
    for layer in vgg16.layers:
      layer.trainable = False

  pool_layer_names = ['block' + str(i) + '_pool' for i in range(1,6)]
  layers = [l for l in output_layers if l in pool_layer_names]
  return Model(inputs=vgg16.input, outputs = [vgg16.get_layer(ln).output
                                                 for ln in layers] )

def vgg16_enc_dec_model_v2(img_shape, reg_dec, outs):
  return dyn_enc_dec_model(img_shape,
                           encoder = vgg16_enc_model(),
                           reg_dec = reg_dec)

def incv3_enc_dec_model(img_shape, reg_dec, outs):
  if outs:
    return dyn_enc_dec_model(img_shape,
                             encoder = incv3_enc_model(wanted_layers=outs),
                             reg_dec = reg_dec)
  else:
    return dyn_enc_dec_model(img_shape,
                             encoder = incv3_enc_model(),
                             reg_dec = reg_dec)

def xception_enc_dec_model(img_shape, reg_dec, outs):
  if outs:
    return dyn_enc_dec_model(img_shape,
                             encoder = xception_enc_model(wanted_layers=outs),
                             reg_dec = reg_dec)
  else:
    return dyn_enc_dec_model(img_shape,
                             encoder = xception_enc_model(),
                             reg_dec = reg_dec)

def resnet_50_enc_dec_model(img_shape, reg_dec, outs):
  if outs:
    return dyn_enc_dec_model(img_shape,
                             encoder = resnet_50_enc_model(wanted_layers=outs),
                             reg_dec = reg_dec)
  else:
    return dyn_enc_dec_model(img_shape,
                             encoder = resnet_50_enc_model(),
                             reg_dec = reg_dec)

models_dict = {'vgg16' : vgg16_enc_dec_model,
               'vgg16_dyn' : vgg16_enc_dec_model_v2,
               'inc_v3' : incv3_enc_dec_model,
               'xception' : xception_enc_dec_model,
               'resnet50' : resnet_50_enc_dec_model}