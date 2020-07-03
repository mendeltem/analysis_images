#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:17:53 2019

@author: DD

quick implementation of a vgg16 encoder decoder unet model

output is distribution (divided by sum over outputs)
"""
import tensorflow as tf

keras = tf.keras
layers = keras.layers
losses = keras.losses
models = keras.models
regularizers = keras.regularizers
Model = keras.models.Model
K = keras.backend
Layer = keras.layers.Layer

class DivByRedSum(Layer):
  """Divide by the reduced sum."""
  def __init__(self, output_dim, **kwargs):
    """Divide by the reduced sum."""
    self.output_dim = output_dim
    super(DivByRedSum, self).__init__(**kwargs)

  def build(self, input_shape):
    super(DivByRedSum, self).build(input_shape)

  def call(self, x):
    return x/tf.reduce_sum(x)

  def get_config(self):
    config = super(DivByRedSum, self).get_config()
    config['output_dim'] = self.output_dim
    return config

  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.output_dim)


def act_w_bn(input_tensor,
             activation_name = 'relu',
             use_batch_normalization = True,
             bn_last = True):
  """activation with batch normalization"""
  if not bn_last is True and use_batch_normalization:
    x = layers.BatchNormalization()(input_tensor)
  else:
    x = input_tensor
  x = layers.Activation(activation_name)(x)
  if bn_last is True and use_batch_normalization:
    x = layers.BatchNormalization()(x)
  return x


def decoder_block(tensor_decode,
                  tensor_concat,
                  num_filters,
                  bn_after_act = True,
                  use_batch_normalization = True,
                  reg = False,
                  dropout = None):
  """Decoder block for the encoder decoder model.

  Arguments:
    decode_tensor: tensor as input for the transpose convolution

    concat_tensor: tensor used to concatenate with the deconvoluted
    tensor. has to fit the deconvoluted tensor.

    num_filters: number of filters / channels for output

    bn_after_act: applies batch normalization after activation. otherwise
    it is applied the other way around.

    use_batch_normalization: If True, uses batch normalization.

    reg: if evaluate as True, uses l2 kernel regularization in decoder with
    parameter reg as a float. Often used value for regularization is 0.001.

    dropout: Use dropout as last layer with the given ratio

  Returns:
    tensor
  """
  x = layers.Conv2DTranspose(num_filters,
                             (2, 2),
                             strides=(2, 2),
                             padding='same')(tensor_decode)

  if not tensor_concat is None:
    x = layers.concatenate([tensor_concat, x], axis=-1)

  x = act_w_bn(x,
               use_batch_normalization = use_batch_normalization,
               bn_last = bn_after_act)

  if reg:
    x = layers.Conv2D(num_filters, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(reg))(x)
  else:
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)

  x = act_w_bn(x,
               use_batch_normalization = use_batch_normalization,
               bn_last = bn_after_act)

  x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)

  x = act_w_bn(x,
               use_batch_normalization = use_batch_normalization,
               bn_last = bn_after_act)

  if dropout:
    x = layers.Dropout(dropout)(x)

  return x

def vgg16_enc_model(freeze=True):
  """Returns a model that has all block pool layers of VGG16 as output.
  """
  output_layers = ['block' + str(i) + '_pool' for i in range(1,6)]
  vgg16 = keras.applications.VGG16(include_top=False, weights='imagenet')

  pool_layer_names = ['block' + str(i) + '_pool' for i in range(1,6)]
  layers = [l for l in output_layers if l in pool_layer_names]

  model = Model(inputs=vgg16.input, outputs = [vgg16.get_layer(ln).output
                                                 for ln in layers] )
  if freeze:
    for layer in model.layers:
      model.trainable = False

  return model

#If this is changed. Change same list in module m_preprocess.py
output_choices = ['softmax', 'div_by_sum_and_sigmoid', 'div_by_red_sum',
                  'sigmoid']

def vgg16_enc_dec_model(img_shape,
                        reg = 0.001,
                        dropout = False,
                        freeze_vgg=True,
                        output_choice = ''):
  """Simple vgg16 based encoding and learned decoding model (UNet).

  Encodes by factor 2**5 and decodes back.

  Arguments:
    img_shape: image shape as (height, width, channels)

    reg: float for regularization in the decoding parts of the network.
    If evaluated as False, no regularization. Otherwise regularization with
    this parameter. Often used is 0.001.

    dropout: choose value between 0 and 1 for dropout on each decoder step.

    output_choice: string to choose last layers and output.

  Returns:
    a tf keras model
  """
  inputs = layers.Input(shape=img_shape)
  # (256,256,3)
  vgg16_encoding = vgg16_enc_model(freeze=freeze_vgg)
  encoder0, encoder1, encoder2, encoder3, encoder4 = vgg16_encoding(inputs)
  # (128,128,64), (64,64,128), (32,32,256), (16,16,512), (8,8,512)
  decoder3 = decoder_block(encoder4, encoder3, 256, reg=reg, dropout=dropout)
  # (16,16,256)
  decoder2 = decoder_block(decoder3, encoder2, 128, reg=reg, dropout=dropout)
  # (32,32,128)
  decoder1 = decoder_block(decoder2, encoder1, 64, reg=reg, dropout=dropout)
  # (64,64,64)
  decoder0 = decoder_block(decoder1, encoder0, 32, reg=reg, dropout=dropout)
  # (128,128,32)
  if output_choice == 'softmax':
    final_up = decoder_block(decoder0,
                             None,
                             32,
                             reg=None,
                             use_batch_normalization=True)
    outputs0 = layers.Conv2D(1, (1, 1), activation='relu')(final_up)
    outputs0 = layers.Reshape((img_shape[0] * img_shape[1],))(outputs0)
    outputs0 = layers.Softmax(axis=1)(outputs0)
    outputs = layers.Reshape((img_shape[0],img_shape[1],1))(outputs0)
    model = Model(inputs=[inputs], outputs=[outputs])
  elif output_choice == 'div_by_sum_and_sigmoid':
    final_up = decoder_block(decoder0,
                         None,
                         32,
                         reg=None,
                         use_batch_normalization=True)

    pre_out_0 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(final_up)
    pre_out_01 = layers.Conv2D(1, (1, 1), activation='relu')(pre_out_0)
    outputs0 = DivByRedSum(pre_out_01.shape)(pre_out_01) #Sum over all outputs is now 1.

    pre_out_1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(final_up)
    outputs1 = layers.Conv2D(1, (1, 1), activation='sigmoid')(pre_out_1)

    model = Model(inputs=[inputs], outputs=[outputs0, outputs1])
  elif output_choice == 'div_by_red_sum':
    final_up = decoder_block(decoder0,
                         None,
                         32,
                         reg=None,
                         use_batch_normalization=True)

    pre_out_0 = layers.Conv2D(1, (1, 1), activation='relu')(final_up)
    outputs = DivByRedSum(outputs.shape)(pre_out_0) #Sum over all outputs is now 1.
    model = Model(inputs=[inputs], outputs=[outputs])
  else: #default case 'sigmoid'
    final_up = decoder_block(decoder0,
                             None,
                             32,
                             reg=None,
                             use_batch_normalization=True)
    # (256,256,32)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(final_up)
    model = Model(inputs=[inputs], outputs=[outputs])
  return model
