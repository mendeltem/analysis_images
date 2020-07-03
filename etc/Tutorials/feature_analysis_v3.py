#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 10:41:02 2018
@author: Temuulen
"""


#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import misc
import re
import exp_library as ex

#Load the Experimental Data (Memory)

file_pictures = "blickdaten/finalresulttab_funcsac_SFC_memory.dat"
all_data = pd.read_table(file_pictures,encoding = "ISO-8859-1")

#Filter for the different Experiment type

#masktype == 0 & maskregion == 0: Kontrollbedingung
exp_control_color = ex.data_cleaning(all_data,1,0,0,0)

#masktype == 1 & maskregion == 1: peripherer Tiefpassfilter
exp_per_tief_color = ex.data_cleaning(all_data,1,1,1,0)

#masktype == 2 & maskregion == 1: peripherer Hochpassfilter
exp_per_hoch_color = ex.data_cleaning(all_data,1,2,1,0)


#masktype == 1 & maskregion == 2: zentraler Tiefpassfilter
exp_zen_tief_color = ex.data_cleaning(all_data,1,1,2,0)

#masktype == 2 & maskregion == 2: zentraler Hochpassfilter
exp_zen_hoch_color = ex.data_cleaning(all_data,1,1,2,0)


#without color
#masktype == 0 & maskregion == 0: Kontrollbedingung
exp_control = ex.data_cleaning(all_data,0,0,0,0)

#masktype == 1 & maskregion == 1: peripherer Tiefpassfilter
exp_per_tief = ex.data_cleaning(all_data,0,1,1,0)

#masktype == 2 & maskregion == 1: peripherer Hochpassfilter
exp_per_hoch = ex.data_cleaning(all_data,0,2,1,0)

#masktype == 1 & maskregion == 2: zentraler Tiefpassfilter
exp_zen_tief = ex.data_cleaning(all_data,0,2,1,0)

#masktype == 2 & maskregion == 2: zentraler Hochpassfilter
exp_zen_hoch = ex.data_cleaning(all_data,0,1,2,0)




file = "created_feature_layers/auto_inception_v3_pic_test_8a472a.p"

list_of_layers,path_of_pics, feat_dict = ex.import_layerdata(file)


data = plt.imread("pic_test/1.png")

with_shape = data.shape[1]

hight_shape = data.shape[0]


import exp_library as ex

list_of_ey_xy = ex.get_eyemovements(exp_control)

activations = ex.get_spatial_activation(list_of_layers,
                                        list_of_ey_xy,
                                        with_shape,
                                        hight_shape)


data = plt.imread("pic_test/1.png")

fovea = 30


data.shape

meanlist = []

for i in range(len(list_of_ey_xy.iloc[0,0])):   
    meanlist.append(
            np.mean( 
                data[int(list_of_ey_xy.iloc[0,0][i])-
                     fovea:int(list_of_ey_xy.iloc[0,0][i])+
                     fovea,int(list_of_ey_xy.iloc[0,1][i])-
                     fovea:int(list_of_ey_xy.iloc[0,1][i])+
                     fovea,0] ))


#for i, n in list_of_ey_xy:
    
x = 300
y = 300

small_data  =np.mean( data[x-fovea:x+fovea,y-fovea:y+fovea,0])


plt.plot(meanlist)


list_of_ey_xy.iloc[0,2][1]

#solutions.append(layer_matrix[i,xs-gx:xs+gx+1,ys-gy:ys+gy+1].mean())




k = [100]

sample_eye = pd.DataFrame([
                          [[100,100],[100,100]],
                          [[100,100],[100,100]],
                          [[100,100],[100,100]],
                          [[100,100],[100,100]]
                          ])



sams = ex.get_spatial_activation(list_of_layers,
                                 sample_eye, 
                                 with_shape, 
                                 hight_shape)
