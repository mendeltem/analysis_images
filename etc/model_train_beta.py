#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:06:17 2019

eye movement predictor, load the model and images from the eye movement
experiment and predict the eye movement with kernel density estimation mask.

@author: mut,dd
"""
import os
#set the working directory
ML_PATH ="/home/mendel/Documents/work/mendel_exchange"
os.chdir(ML_PATH)
#from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow import keras
ModelCheckpoint = keras.callbacks.ModelCheckpoint
#ModelCheckpoint = keras.callbacks.EarlyStopping


from modules.model_zoo_alpha import models_dict
from modules.library import plot_training_history,evaluate,evaluate_model
from m_model import keras_model, keras_cmodel

import subprocess
from modules.model_zoo import bce_loss, dice_loss, bce_dice_loss
from modules.model_zoo import precision, recall

import tensorflow as tf
from m_model import keras_model, keras_cmodel,keras_cmodel_beta,keras_bmodel
import sys

from keras.optimizers import SGD, RMSprop
#monitor
my_dpi = 90
#path length for ploting kde images
pad_length = 27
#widh and height
#w = 1024+pad_length
#h = 768+pad_length

#saving the model path
SAVE_PATH = 'figures/memory_amodel_weights.hdf5'

#one experiment type path
TRAIN_COLOR_ORIGINAL_MEMORY      = os.getcwd()+"/dataset/color_memory/original/train"
TEST_COLOR_ORIGINAL_MEMORY       = os.getcwd()+"/dataset/color_memory/original/test"
#one experiment type path
TRAIN_COLOR_PER_HOCH_MEMORY      = os.getcwd()+"/dataset/color_memory/periphererHochpass/train"
TEST_COLOR_PER_HOCH_MEMORY       = os.getcwd()+"/dataset/color_memory/periphererHochpass/test"
#one experiment type path
TRAIN_COLOR_PER_TIEF_MEMORY      = os.getcwd()+"/dataset/color_memory/periphererTiefpass/train"
TEST_COLOR_PER_TIEF_MEMORY       = os.getcwd()+"/dataset/color_memory/periphererTiefpass/test"
#one experiment type path
TRAIN_COLOR_ZEN_HOCH_MEMORY      = os.getcwd()+"/dataset/color_memory/zentralerHochpass/train"
TEST_COLOR_ZEN_HOCH_MEMORY       = os.getcwd()+"/dataset/color_memory/zentralerHochpass/test"
#one experiment type path
TRAIN_COLOR_ZEN_TIEF_MEMORY      = os.getcwd()+"/dataset/color_memory/zentralerTiefpass/train"
TEST_COLOR_ZEN_TIEF_MEMORY       = os.getcwd()+"/dataset/color_memory/zentralerTiefpass/test"


#one experiment type path
TRAIN_COLOR_ORIGINAL_SEARCH       = os.getcwd()+"/dataset/grayscale_memory/original/train"
TEST_COLOR_ORIGINAL_SEARCH        = os.getcwd()+"/dataset/grayscale_memory/original/test"
#one experiment type path
TRAIN_COLOR_PER_HOCH_SEARCH       = os.getcwd()+"/dataset/grayscale_memory/periphererHochpass/train"
TEST_COLOR_PER_HOCH_SEARCH        = os.getcwd()+"/dataset/grayscale_memory/periphererHochpass/test"
#one experiment type path
TRAIN_COLOR_PER_TIEF_SEARCH       = os.getcwd()+"/dataset/grayscale_memory/periphererTiefpass/train"
TEST_COLOR_PER_TIEF_SEARCH        = os.getcwd()+"/dataset/grayscale_memory/periphererTiefpass/test"
#one experiment type path
TRAIN_COLOR_ZEN_HOCH_SEARCH       = os.getcwd()+"/dataset/grayscale_memory/zentralerHochpass/train"
TEST_COLOR_ZEN_HOCH_SEARCH        = os.getcwd()+"/dataset/grayscale_memory/zentralerHochpass/test"
#one experiment type path
TRAIN_COLOR_ZEN_TIEF_SEARCH       = os.getcwd()+"/dataset/grayscale_memory/zentralerTiefpass/train"
TEST_COLOR_ZEN_TIEF_SEARCH        = os.getcwd()+"/dataset/grayscale_memory/zentralerTiefpass/test"

#experiment type path
TRAIN_GRAY_MEMORY       = os.getcwd()+"/dataset/grayscale_memory/original/train"
TEST_GRAY_MEMORY        = os.getcwd()+"/dataset/grayscale_memory/original/test"
#all experiment in one folder
ALL_TRAIN_PATH = os.getcwd()+"/dataset/training"
ALL_TEST_PATH = os.getcwd()+"/dataset/validation"
#collection of 300 pictures from search
TRAIN_PATH_SEARCH   = os.getcwd()+"/dataset/both_original_perHochpass_search/train"
TEST_PATH_SEARCH   = os.getcwd()+"/dataset/both_original_perHochpass_search/test"
#collection of 300 pictures from memory
TRAIN_PATH_MEMORY   = os.getcwd()+"/dataset/both_original_perHochpass_memory/train"
TEST_PATH_MEMORY   = os.getcwd()+"/dataset/both_original_perHochpass_memory/test"
#change here

#New Data Set
#experiment type path
BIG_DATA_TRAIN      = os.getcwd()+"/dataset/Set_01/train"
BIG_DATA_TEST        = os.getcwd()+"/dataset/Set_01/test"

PATH_TRAIN  = BIG_DATA_TRAIN
PATH_TEST   = BIG_DATA_TEST

w = 1024
h = 768
BATCH_SIZE = 32

from modules.m_preprocess import image_mask_generator,load_images_and_masks
#from modules.Image_Mask_DataGenerator import Image_DataGenerator
# Set parameters for Image base Size
IMG_WIDTH =  1024
IMG_HEIGHT = 768
IMG_CHANNELS = 3
#downsize the images and masks
down_size =  4
#Training Bachsize

#random seed
seed = 42
Epochs = 100

if  len(sys.argv) > 1:
  Epochs = sys.argv

img_height = int(IMG_HEIGHT/down_size)
img_width  = int(IMG_WIDTH/down_size)


#load the images and masks and perform pooling before resize again
trainX,trainY = load_images_and_masks(PATH_TRAIN, img_width=img_width, img_height=img_height, method = "mean",pool_size = 1)
testX,testY   = load_images_and_masks(PATH_TEST,  img_width=img_width, img_height=img_height, method = "mean",pool_size = 1)
valX,valY     = load_images_and_masks(TRAIN_COLOR_ORIGINAL_MEMORY,  img_width=img_width, img_height=img_height, method = "mean",pool_size = 1)

#Generator
train_generator,test_generator =  image_mask_generator(trainX,
                                                       trainY,
                                                       testX,
                                                       testY,
                                                       BATCH_SIZE)



#------------------------------------------------------------------------------------------

#workin on cmodel
cmodel = keras_cmodel_beta(img_height,img_width)

#opt = SGD(lr = 1e-2, momentum = 0.9, nesterov = True)
#cmodel_beta.compile(optimizer = opt, loss = 'mse')


cmodel_dir = 'figures/cmodel/'
save_cmodel_wieghts_path = cmodel_dir +'model_weights_best.hdf5'

subprocess.call("mkdir "+cmodel_dir , shell = True )
subprocess.call("mkdir "+cmodel_dir+"test/" , shell = True )
subprocess.call("mkdir "+cmodel_dir+"train/", shell = True )
subprocess.call("mkdir "+cmodel_dir+"val/", shell = True )

#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

#training the model
history= cmodel.fit_generator(
           train_generator,
           validation_data=test_generator,
           validation_steps=BATCH_SIZE/2,
           steps_per_epoch=len(trainX)/(BATCH_SIZE*2),
           callbacks=[ ModelCheckpoint(
               filepath=save_cmodel_wieghts_path,
               verbose=1, save_best_only=True)],
           epochs=Epochs).history

with open(cmodel_dir+"trainHistoryDict", 'wb') as file_pi:
    pickle.dump(history, file_pi)

history = pickle.load( open( cmodel_dir+"trainHistoryDict", "rb" ) )

plt.loglog(history["val_dice_loss"], linestyle='-', color='g', label='val_dice_loss: ' +str(round(history["val_dice_loss"][-1], 3)))
plt.loglog(history["dice_loss"], linestyle='-', color='y', label='dice_loss: ' +str(round(history["dice_loss"][-1], 3)))
plt.loglog(history["val_loss"], linestyle='-', color='r', label='Val Loss: ' +str(round(history["val_loss"][-1], 3)))
plt.loglog(history["loss"], linestyle='-', color='b', label='Loss: ' +str(round(history["loss"][-1], 3)))
plt.title('Training and Test Loss')
plt.legend()
plt.savefig(cmodel_dir+"history.jpg")
plt.show()

#plot_training_history(history,save_dir=cmodel_dir, plotname="cmodel_history")
cmodel.load_weights(save_cmodel_wieghts_path)
#testing on test set
#y = cmodel.predict(testX)
#evaluate(testX,y,testY, threshhold=0.05,save_dir = cmodel_dir,history= history,index=1)
#evaluate(testX,y,testY, threshhold=0.05,save_dir = cmodel_dir,history= history,index=2)
#evaluate(testX,y,testY, threshhold=0.05,save_dir = cmodel_dir,history= history,index=3)
#evaluate(testX,y,testY, threshhold=0.05,save_dir = cmodel_dir,history= history,index=4)
#
#plt.imshow(testX[10])
#
#plt.imshow(np.squeeze(testY[10]))
#
#plt.imshow(np.squeeze(y[10]))
#
#
#
#real = np.squeeze(y[10])
#pred = np.squeeze(y[10])
#
#fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15,15))
#plt.subplot(2, 1, 1)
#
#intersect = real * pred
#plt.title("Intersect")
#plt.imshow(intersect)
#plt.subplot(2, 1, 2)
#
#union =  np.or(real, pred)
#plt.title("Union")
#plt.imshow(union)
#
#
#prob = np.sum(intersect)/np.sum(union)


#
#accuracy,precision,recall = evaluate_model(pred = y,labels = testY,threshhold=0.01)
#
#print("mean accuracy: ",np.mean(accuracy))
#print("mean precision: ",np.mean(precision))
#print("mean recall: ",np.mean(recall))
#
#evaluate(testX,y,testY, threshhold=0.05,save_dir = cmodel_dir,history= history,index=10)
#
##test plot
#for im in range(len(y)):
#  if im > 45: break
#  plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
#  plt.imshow(testX[im])
#  plt.savefig( cmodel_dir+"test/"+str(im)+"_image_test_out.png")
#  plt.imshow(np.squeeze(testY[im]))
#  plt.savefig( cmodel_dir+"test/"+str(im)+"_mask_a_target_out.png")
#  plt.imshow(np.squeeze(y[im]))
#  plt.savefig( cmodel_dir+"test/"+str(im)+"_mask_b_predict_out.png")
#
##testing on train set
#y_train = cmodel.predict(trainX)
#

#evaluate(trainX,y_train,y_train, threshhold=0.05,save_dir = cmodel_dir,history= history,index=20)

#
#
#
#
#evaluate(trainX,y_train,trainY, threshhold=0.05,save_dir = cmodel_dir,history= history,index=1)
#evaluate(trainX,y_train,trainY, threshhold=0.05,save_dir = cmodel_dir,history= history,index=2)
#evaluate(trainX,y_train,trainY, threshhold=0.05,save_dir = cmodel_dir,history= history,index=3)
#evaluate(trainX,y_train,trainY, threshhold=0.05,save_dir = cmodel_dir,history= history,index=4)
#
#
#
#
#
#
#for im in range(len(y_train)):
#  if im > 93: break
#  plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
#  plt.imshow(trainX[im])
#  plt.savefig( cmodel_dir+"train/"+str(im)+"_image_train_out.png")
#  plt.imshow(np.squeeze(trainY[im]))
#  plt.savefig( cmodel_dir+"train/"+str(im)+"_mask_a_target_out.png")
#  plt.imshow(np.squeeze(y_train[im]))
#  plt.savefig(cmodel_dir+"train/"+str(im)+"_mask_b_predict_out.png")
#
##testing on train set
#y_val = cmodel.predict(valX)
#
#evaluate(valX,y_val,valY, threshhold=0.05,save_dir = cmodel_dir,history= history,index=1)
#evaluate(valX,y_val,valY, threshhold=0.05,save_dir = cmodel_dir,history= history,index=2)
#evaluate(valX,y_val,valY, threshhold=0.05,save_dir = cmodel_dir,history= history,index=3)
#evaluate(valX,y_val,valY, threshhold=0.05,save_dir = cmodel_dir,history= history,index=4)
#
#
#evaluate(valX,y_val,valY, threshhold=0.05,save_dir = cmodel_dir,history= history,index=7)
#
#np.max(y_val)
#
#
#
#
#for im in range(len(valX)):
#  plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
#  plt.imshow(valX[im] )
#  plt.savefig( cmodel_dir+"val/"+str(im)+"_image_train_out.png")
#  plt.imshow(np.squeeze(valY[im]))
#  plt.savefig( cmodel_dir+"val/"+str(im)+"_mask_a_target_out.png")
#  plt.imshow(np.squeeze(y_val[im]))
#  plt.savefig(cmodel_dir+"val/"+str(im)+"_mask_b_predict_out.png")
#  #-----------------------------------------------------------------------------------
#
#
##workin on amodel
#amodel = models_dict['vgg16']((int(IMG_HEIGHT / down_size),
#                    int(IMG_WIDTH/ down_size),3), False, None)
#
#
#loss = 'bce_dice_loss'
#metrics = ['dice_loss']
#
##dict to make variable selection simpler
#metrics_and_losses = {'dice_loss':dice_loss,
#                       'bce_dice_loss':bce_dice_loss,
#                       'bce_loss':bce_loss,
#                       'precision': precision,
#                       'recall': recall}
#
#
#amodel.compile(optimizer='adam',
#            loss=metrics_and_losses[loss],
#            metrics=[metrics_and_losses[m] for m in metrics])
#
#
#amodel_dir = 'figures/amodel/'
#save_amodel_wieghts_path = amodel_dir +'model_weights_best.hdf5'
#
#
#subprocess.call("mkdir "+amodel_dir , shell = True )
#subprocess.call("mkdir "+amodel_dir+"test/" , shell = True )
#subprocess.call("mkdir "+amodel_dir+"train/", shell = True )
#subprocess.call("mkdir "+amodel_dir+"val/", shell = True )
#
##training the model
#history= amodel.fit_generator(
#           train_generator,
#           validation_data=test_generator,
#           validation_steps=BATCH_SIZE/2,
#           steps_per_epoch=len(trainX)/(BATCH_SIZE*2),
#           callbacks=[es, ModelCheckpoint(
#               filepath=save_amodel_wieghts_path,
#               verbose=1, save_best_only=True)],
#           epochs=1000).history
#
#
#with open(amodel_dir+"trainHistoryDict", 'wb') as file_pi:
#    pickle.dump(history, file_pi)
#
#
#history = pickle.load( open( amodel_dir+"trainHistoryDict", "rb" ) )
#
#plt.loglog(history["val_loss"], linestyle='-', color='r', label='Val Loss: ' +str(round(history["val_loss"][-1], 3)))
#plt.loglog(history["loss"], linestyle='-', color='b', label='Loss: ' +str(round(history["loss"][-1], 3)))
#plt.title('Training and Test Loss')
#plt.legend()
#plt.savefig(amodel_dir+"history.jpg")
#plt.show()
#
#
#amodel.load_weights(save_amodel_wieghts_path)
#
##testing on test set
#y = amodel.predict(testX)
#
#evaluate(testX,y,testY, threshhold=0.05,save_dir = amodel_dir+"1",history= history,index=1)
#evaluate(testX,y,testY, threshhold=0.05,save_dir = amodel_dir+"2",history= history,index=2)
#evaluate(testX,y,testY, threshhold=0.05,save_dir = amodel_dir+"3",history= history,index=3)
#evaluate(testX,y,testY, threshhold=0.05,save_dir = amodel_dir+"4",history= history,index=4)
#
##test plot
#for im in range(len(y)):
#  if im > 60: break
#  plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
#  plt.imshow(testX[im])
#  plt.savefig( amodel_dir+"test/"+str(im)+"_image_test_out.png")
#  plt.imshow(np.squeeze(testY[im]))
#  plt.savefig( amodel_dir+"test/"+str(im)+"_mask_a_target_out.png")
#  plt.imshow(np.squeeze(y[im]))
#  plt.savefig( amodel_dir+"test/"+str(im)+"_mask_b_predict_out.png")
#
##testing on train set
#y_train = amodel.predict(trainX)
#np.max(trainY)
#
#evaluate(trainX,y_train,trainY, threshhold=0.05,save_dir = amodel_dir+"1_train",history= history,index=1)
#evaluate(trainX,y_train,trainY, threshhold=0.05,save_dir = amodel_dir+"2_train",history= history,index=2)
#evaluate(trainX,y_train,trainY, threshhold=0.05,save_dir = amodel_dir+"3_train",history= history,index=3)
#evaluate(trainX,y_train,trainY, threshhold=0.05,save_dir = amodel_dir+"4_train",history= history,index=4)
#
#for im in range(len(y_train)):
#  if im > 93: break
#  plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
#  plt.imshow(trainX[im] )
#  plt.savefig( amodel_dir+"train/"+str(im)+"_image_train_out.png")
#  plt.imshow(np.squeeze(trainY[im]))
#  plt.savefig( amodel_dir+"train/"+str(im)+"_mask_a_target_out.png")
#  plt.imshow(np.squeeze(y_train[im]))
#  plt.savefig(amodel_dir+"train/"+str(im)+"_mask_b_predict_out.png")
#
##testing on validation set
#y_val = amodel.predict(valX)
#
#evaluate(valX,y_val,valY, threshhold=0.05,save_dir = amodel_dir+"1_val",history= history,index=1)
#evaluate(valX,y_val,valY, threshhold=0.05,save_dir = amodel_dir+"2_val",history= history,index=2)
#evaluate(valX,y_val,valY, threshhold=0.05,save_dir = amodel_dir+"3_val",history= history,index=3)
#evaluate(valX,y_val,valY, threshhold=0.05,save_dir = amodel_dir+"4_val",history= history,index=4)
#
#for im in range(len(valX)):
#  plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
#  plt.imshow(valX[im] )
#  plt.savefig( amodel_dir+"val/"+str(im)+"_image_train_out.png")
#  plt.imshow(np.squeeze(valY[im]))
#  plt.savefig( amodel_dir+"val/"+str(im)+"_mask_a_target_out.png")
#  plt.imshow(np.squeeze(y_val[im]))
#  plt.savefig(amodel_dir+"val/"+str(im)+"_mask_b_predict_out.png")
#
#
##-----------------------------------------------------------------------------------------------------------------------
##BATCH_SIZE = 8
###Generator
##train_generator,test_generator =  image_mask_generator(trainX,
##                                                       trainY,
##                                                       testX,
##                                                       testY,
##                                                       BATCH_SIZE)
##
###workin on amodel that resembles Deepgaze
##bmodel = keras_bmodel(img_height,img_width)
##
##bmodel_dir = 'figures/bmodel/'
##save_bmodel_wieghts_path = bmodel_dir +'model_weights_best.hdf5'
##
##
##subprocess.call("mkdir "+bmodel_dir , shell = True )
##subprocess.call("mkdir "+bmodel_dir+"test/" , shell = True )
##subprocess.call("mkdir "+bmodel_dir+"train/", shell = True )
##subprocess.call("mkdir "+bmodel_dir+"val/", shell = True )
##
###training the model
##history= bmodel.fit_generator(
##           train_generator,
##           validation_data=test_generator,
##           validation_steps=BATCH_SIZE/2,
##           steps_per_epoch=len(trainX)/(BATCH_SIZE*2),
##           callbacks=[ ModelCheckpoint(
##               filepath=save_bmodel_wieghts_path,
##               verbose=1, save_best_only=True)],
##           epochs=2).history
##
##
##with open(bmodel_dir+"trainHistoryDict", 'wb') as file_pi:
##    pickle.dump(history, file_pi)
##
##
##history = pickle.load( open( bmodel_dir+"trainHistoryDict", "rb" ) )
##
##plt.loglog(history["val_loss"], linestyle='-', color='r', label='Val Loss: ' +str(round(history["val_loss"][-1], 3)))
##plt.loglog(history["loss"], linestyle='-', color='b', label='Loss: ' +str(round(history["loss"][-1], 3)))
##plt.title('Training and Test Loss')
##plt.legend()
##plt.savefig(bmodel_dir+"history.jpg")
##plt.show()
##
##
##bmodel.load_weights(save_bmodel_wieghts_path)
##
###testing on test set
##y = bmodel.predict(testX) #test plot
##evaluate(testX,y,testY, threshhold=0.10,save_dir = bmodel_dir,history= history,index=5)
##
##for im in range(len(y)):
##  if im > 5: break
##  plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
##  plt.imshow(testX[im])
##  plt.savefig( bmodel_dir+"test/"+str(im)+"_image_test_out.png")
##  plt.imshow(np.squeeze(testY[im]))
##  plt.savefig( bmodel_dir+"test/"+str(im)+"_mask_a_target_out.png")
##  plt.imshow(np.squeeze(y[im]))
##  plt.savefig( bmodel_dir+"test/"+str(im)+"_mask_b_predict_out.png")
##
###testing on train set
##y_train = bmodel.predict(trainX)
##
##for im in range(len(y_train)):
##  if im > 5: break
##  plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
##  plt.imshow(trainX[im] )
##  plt.savefig( bmodel_dir+"train/"+str(im)+"_image_train_out.png")
##  plt.imshow(np.squeeze(trainY[im]))
##  plt.savefig( bmodel_dir+"train/"+str(im)+"_mask_a_target_out.png")
##  plt.imshow(np.squeeze(y_train[im]))
##  plt.savefig(bmodel_dir+"train/"+str(im)+"_mask_b_predict_out.png")
##
###testing on validation set
##y_val = bmodel.predict(valX)
##
##for im in range(len(valX)):
##  if im > 5: break
##  plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
##  plt.grid(False)
##  plt.imshow(valX[im] )
##  plt.savefig( bmodel_dir+"val/"+str(im)+"_image_train_out.png")
##  plt.grid(False)
##  plt.imshow(np.squeeze(valY[im]))
##  plt.savefig( bmodel_dir+"val/"+str(im)+"_mask_a_target_out.png")
##  plt.grid(False)
##  plt.imshow(np.squeeze(y_val[im]))
##  plt.savefig(bmodel_dir+"val/"+str(im)+"_mask_b_predict_out.png")
##
#
##sys.exit()