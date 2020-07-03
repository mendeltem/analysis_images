#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 15:17:26 2018

@author: DD, UT

compute and save intermediate representations of fixations on images from the
inception_v3 net.
"""

import sys
import pickle
import keras
import os
import gc
#import exp_library as ex
import pandas as pd
import numpy as np
import re
from time import time


DEFAULT_EYE_FIXATION_DAT    = "finalresulttab_funcsac_SFC_memory.dat"
DEFAULT_IMAGE_DIRECTORY     = 'mem_images/'
DEFAULT_SAVE_DATA_PATH      = 'activations_of_fixations_auto.p' #TODO: think about a good name maybe
DEFAULT_EYE_FIXATION_DAT_MEMORY    = "finalresulttab_funcsac_SFC_memory.dat"

DEFAULT_EYE_FIXATION_DAT_SEARCH    = "finalresulttab_funcsac_SFC_search.dat"


all_data_uncleared = pd.read_table(DEFAULT_EYE_FIXATION_DAT,encoding = "ISO-8859-1")

all_data_search_uncleared = pd.read_table(DEFAULT_EYE_FIXATION_DAT_SEARCH,encoding = "ISO-8859-1")


SAVE_PATH_MODEL = ''
IMAGE_DIRECTORY = ''
SAVE_DATA_PATH = ''


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

if IMAGE_DIRECTORY == '':
    img_paths = get_img_paths(DEFAULT_IMAGE_DIRECTORY)
else:
    img_paths = get_img_paths(IMAGE_DIRECTORY)

#set the fovea size
fovea= 30
#filter: get only the fixation part inside the maximum fovea possible

all_data = all_data_uncleared.loc[
                      (all_data_uncleared["fixposx"] >= fovea) &
                      (all_data_uncleared["fixposx"] <= 1024 - fovea) &
                      (all_data_uncleared["fixposy"] >= fovea) &
                      (all_data_uncleared["fixposy"] <= 768 - fovea) & 
                       (all_data_uncleared["fixinvalid"] != 1) &
                       (all_data_uncleared["sacinvalid"] != 1)
                      ]






#filter: get only the fixation part inside the maximum fovea possible
all_data_search = all_data_search_uncleared.loc[
                      (all_data_search_uncleared["fixposx"] >= fovea) &
                      (all_data_search_uncleared["fixposx"] <= 1024 - fovea) &
                      (all_data_search_uncleared["fixposy"] >= fovea) &
                      (all_data_search_uncleared["fixposy"] <= 768 - fovea) &
                       (all_data_search_uncleared["fixinvalid"] != 1) &
                       (all_data_search_uncleared["sacinvalid"] != 1)
                      ]


#ungereinigte Datenmenge
print("ungereinigte Datenmenge Memory ",all_data_uncleared.shape[0])   
 
#gereinigte Datenmenge   
print("gereinigte Datenmenge Memory ",fovea ,"fovea größe ",all_data.shape[0])   


#ungereinigte Datenmenge
print("ungereinigte Datenmenge Search ",all_data_search_uncleared.shape[0])   
 
#gereinigte Datenmenge   
print("gereinigte Datenmenge Search ",fovea ,"fovea größe ",all_data_search.shape[0])  


#All_data
alle_data = all_data.loc[ ((all_data["colorimages"] == 1) | (all_data["colorimages"] == 0) )&
                        ((all_data["masktype"] == 0) | (all_data["masktype"] == 1) | (all_data["masktype"] == 2)) &
                        ((all_data["maskregion"]  == 1 ) | (all_data["maskregion"]  == 0 )  | (all_data["maskregion"]  == 2 ) )
                        ,:] 

print("Alle Daten", alle_data.shape[0])


#        Daraus ergibt sich entsprechend:
#        masktype == 0 & maskregion == 0: Kontrollbedingung
#        masktype == 1 & maskregion == 1: peripherer Tiefpassfilter
#        masktype == 2 & maskregion == 1: peripherer Hochpassfilter
#        masktype == 1 & maskregion == 2: zentraler Tiefpassfilter
#        masktype == 2 & maskregion == 2: zentraler Hochpassfilter

     



#        masktype == 0 & maskregion == 0: Kontrollbedingung
#alle Kontrollbedingung ohne Farben
controll_data = all_data.loc[ ((all_data["colorimages"] == 1) | (all_data["colorimages"] == 0) )&
                        ((all_data["masktype"] == 0)) &
                        ( (all_data["maskregion"]  == 0 ) )
                        ,:] 

print("Keine/Mit Farben Kontrollbedingung", controll_data.shape[0])

#mit Farben
controll_data_color = all_data.loc[ ((all_data["colorimages"] == 1) )&
                        ((all_data["masktype"] == 0)) &
                        ( (all_data["maskregion"]  == 0 ) )
                        ,:] 

print("Mit Farben Kontrollbedingung", controll_data_color.shape[0])
#keine Farben
controll_data_grey = all_data.loc[ ((all_data["colorimages"] == 0) )&
                        ((all_data["masktype"] == 0)) &
                        ( (all_data["maskregion"]  == 0 ) )
                        ,:] 
print("Keine Farben Kontrollbedingung", controll_data_grey.shape[0])



#alle

#        masktype == 1 & maskregion == 1: peripherer Tiefpassfilter
# alle
per_Tief_data = all_data.loc[ ((all_data["colorimages"] == 1) | (all_data["colorimages"] == 0) )&
                        ((all_data["masktype"] == 1) ) &
                        ((all_data["maskregion"]  == 1 ) )
                        ,:] 

print("Keine/Mit Farben peripherer Tiefpassfilter", per_Tief_data.shape[0])

# mit Farben
per_Tief_data_color = all_data.loc[ ((all_data["colorimages"] == 1)  )&
                        ((all_data["masktype"] == 1) ) &
                        ((all_data["maskregion"]  == 1 ) )
                        ,:] 

print("Mit Farben peripherer Tiefpassfilter", per_Tief_data_color.shape[0])

# mit Farben
per_Tief_data_grey = all_data.loc[ ((all_data["colorimages"] == 0)  )&
                        ((all_data["masktype"] == 1) ) &
                        ((all_data["maskregion"]  == 1 ) )
                        ,:] 

print("Ohne Farben peripherer Tiefpassfilter", per_Tief_data_grey.shape[0])


#        masktype == 2 & maskregion == 1: peripherer Hochpassfilter
# alle
per_Hoch_data = all_data.loc[ ((all_data["colorimages"] == 1) | (all_data["colorimages"] == 0) )&
                        ((all_data["masktype"] == 2) ) &
                        ((all_data["maskregion"]  == 1 ) )
                        ,:] 

print("Keine/Mit Farben peripherer Hochpassfilter", per_Hoch_data.shape[0])

# mit Farben
per_Hoch_data_color = all_data.loc[ ((all_data["colorimages"] == 1)  )&
                        ((all_data["masktype"] == 2) ) &
                        ((all_data["maskregion"]  == 1 ) )
                        ,:] 

print("Mit Farben peripherer Hochpassfilter", per_Hoch_data_color.shape[0])

# mit Farben
per_Hoch_data_grey = all_data.loc[ ((all_data["colorimages"] == 0)  )&
                        ((all_data["masktype"] == 2) ) &
                        ((all_data["maskregion"]  == 1 ) )
                        ,:] 

print("Ohne Farben peripherer Hochpassfilter", per_Hoch_data_grey.shape[0])



#        masktype == 1 & maskregion == 2: zentraler Tiefpassfilter

# alle
zen_Tief_data = all_data.loc[ ((all_data["colorimages"] == 1) | (all_data["colorimages"] == 0) )&
                        ((all_data["masktype"] == 2) ) &
                        ((all_data["maskregion"]  == 2) )
                        ,:] 

print("Keine/Mit Farben zentraler Tiefpassfilter", zen_Tief_data.shape[0])

# mit Farben
zen_Tief_data_color = all_data.loc[ ((all_data["colorimages"] == 1)  )&
                        ((all_data["masktype"] == 1) ) &
                        ((all_data["maskregion"]  == 2 ) )
                        ,:] 

print("Mit Farben zentraler Tiefpassfilter", zen_Tief_data_color.shape[0])

# Ohne Farben
zen_Tief_data_grey = all_data.loc[ ((all_data["colorimages"] == 0)  )&
                        ((all_data["masktype"] == 1) ) &
                        ((all_data["maskregion"]  == 2 ) )
                        ,:] 

print("Ohne Farben zentraler Tiefpassfilter", zen_Tief_data_grey.shape[0])



#        masktype == 2 & maskregion == 2: zentraler Hochpassfilter
# alle
zen_Hoch_data = all_data.loc[ ((all_data["colorimages"] == 1) | (all_data["colorimages"] == 0) )&
                        ((all_data["masktype"] == 2) ) &
                        ((all_data["maskregion"]  == 2) )
                        ,:] 

print("Keine/Mit Farben zentraler Hochpassfilter", zen_Hoch_data.shape[0])

# mit Farben
zen_Hoch_data_color = all_data.loc[ ((all_data["colorimages"] == 1)  )&
                        ((all_data["masktype"] == 2) ) &
                        ((all_data["maskregion"]  == 2 ) )
                        ,:] 

print("Mit Farben zentraler Hochpassfilter", zen_Hoch_data_color.shape[0])

# Ohne Farben
zen_Hoch_data_grey = all_data.loc[ ((all_data["colorimages"] == 0)  )&
                        ((all_data["masktype"] == 2) ) &
                        ((all_data["maskregion"]  == 2 ) )
                        ,:] 

print("Ohne Farben zentraler Hochpassfilter", zen_Hoch_data_grey.shape[0])



zentrales = pd.DataFrame()
zentrales = zentrales.append(zen_Tief_data_color)
zentrales = zentrales.append(zen_Tief_data_grey)
zentrales = zentrales.append(zen_Hoch_data_color)
zentrales = zentrales.append(zen_Hoch_data_grey)
#
#zentrales = zentrales.sort_values(by=['imageid'])
#  
#Datei_t = zentrales.loc[(zentrales["imageid"]== 1)& (Output_df["subject"]==3), :]
#
#
#Datei_t = Datei_t.sort_values(by=['subject',"fixno"])
    

control  = controll_data_color.shape[0] + controll_data_grey.shape[0] 
perTief  = per_Tief_data_color.shape[0] + per_Tief_data_grey.shape[0]
perHoch  = per_Hoch_data_color.shape[0] + per_Hoch_data_grey.shape[0]
zentTief = zen_Tief_data_color.shape[0] + zen_Tief_data_grey.shape[0]
zenHoch  = zen_Hoch_data_color.shape[0] + zen_Hoch_data_grey.shape[0]


print("Kontrollbedingung ",control)
print(" peripherer Tiefpassfilter ",perTief)
print(" peripherer Hochpassfilter ",perHoch)
print(" zentraler Tiefpassfilter ",zentTief)
print(" zentraler Hochpassfilter ",zenHoch)

gesammt = control+ perTief +perHoch +zentTief +zenHoch
print(gesammt)

zentTief+zenHoch

per = control+ perTief +perHoch 
print(per)




number =80

selected_data_o = all_data.loc[ 
    (all_data["imageid"] == int(number))&
    ((all_data["colorimages"] == 1) | (all_data["colorimages"] == 0) )&
    ((all_data["masktype"] == 0) | (all_data["masktype"] == 1)
    | (all_data["masktype"] == 2) ) &
    ( (all_data["maskregion"]  == 0 | (all_data["maskregion"] == 1) ) )
    ,:] 

selected_data_h = all_data.loc[ 
      (all_data["imageid"] == int(number))&
      ((all_data["colorimages"] == 1) | (all_data["colorimages"] == 0) )&
      ((all_data["masktype"] == 2) ) &
      ((all_data["maskregion"]  == 2) )
        ,:] 


#Seach----------------------------------------------------------------------------------------------------------------------

#        masktype == 0 & maskregion == 0: Kontrollbedingung
#alle Kontrollbedingung ohne Farben
controll_data_search = all_data_search.loc[ 
        ((all_data_search["colorimages"] == 1)
        | (all_data_search["colorimages"] == 0) )&
        ((all_data_search["masktype"] == 0)) &
        ( (all_data_search["maskregion"]  == 0 ) )
        ,:] 

print("Keine/Mit Farben Kontrollbedingung", controll_data_search.shape[0])

#mit Farben
controll_data_color_search = all_data_search.loc[ 
        ((all_data_search["colorimages"] == 1) )&
        ((all_data_search["masktype"] == 0)) &
        ( (all_data_search["maskregion"]  == 0 ) )
        ,:] 

print("Mit Farben Kontrollbedingung", controll_data_color_search.shape[0])
#keine Farben
controll_data_grey_search = all_data_search.loc[ 
        ((all_data_search["colorimages"] == 0) )&
        ((all_data_search["masktype"] == 0)) &
        ( (all_data_search["maskregion"]  == 0 ) )
        ,:] 
print("Keine Farben Kontrollbedingung", controll_data_grey_search.shape[0])



#alle

#        masktype == 1 & maskregion == 1: peripherer Tiefpassfilter
# alle
per_Tief_data_search = all_data_search.loc[ 
        ((all_data_search["colorimages"] == 1)
        | (all_data_search["colorimages"] == 0) )&
        ((all_data_search["masktype"] == 1) ) &
        ((all_data_search["maskregion"]  == 1 ) )
        ,:] 

print("Keine/Mit Farben peripherer Tiefpassfilter", 
      per_Tief_data_search.shape[0])

# mit Farben
per_Tief_data_color_search = all_data_search.loc[ 
        ((all_data_search["colorimages"] == 1)  )&
        ((all_data_search["masktype"] == 1) ) &
        ((all_data_search["maskregion"]  == 1 ) )
        ,:] 

print("Mit Farben peripherer Tiefpassfilter", 
      per_Tief_data_color_search.shape[0])

# mit Farben
per_Tief_data_grey_search = all_data_search.loc[ 
        ((all_data_search["colorimages"] == 0)  )&
        ((all_data_search["masktype"] == 1) ) &
        ((all_data_search["maskregion"]  == 1 ) )
        ,:] 

print("Ohne Farben peripherer Tiefpassfilter", 
      per_Tief_data_grey_search.shape[0])


#        masktype == 2 & maskregion == 1: peripherer Hochpassfilter
# alle
per_Hoch_data_search = all_data_search.loc[ 
        ((all_data_search["colorimages"] == 1)
        | (all_data_search["colorimages"] == 0) )&
        ((all_data_search["masktype"] == 2) ) &
        ((all_data_search["maskregion"]  == 1 ) )
        ,:] 

print("Keine/Mit Farben peripherer Hochpassfilter", 
      per_Hoch_data_search.shape[0])

# mit Farben
per_Hoch_data_color_search = all_data_search.loc[
        ((all_data_search["colorimages"] == 1)  )&
        ((all_data_search["masktype"] == 2) ) &
        ((all_data_search["maskregion"]  == 1 ) )
        ,:] 

print("Mit Farben peripherer Hochpassfilter", 
      per_Hoch_data_color_search.shape[0])

# mit Farben
per_Hoch_data_grey_search = all_data_search.loc[ 
        ((all_data_search["colorimages"] == 0)  )&
        ((all_data_search["masktype"] == 2) ) &
        ((all_data_search["maskregion"]  == 1 ) )
        ,:] 

print("Ohne Farben peripherer Hochpassfilter", 
      per_Hoch_data_grey_search.shape[0])



#        masktype == 1 & maskregion == 2: zentraler Tiefpassfilter

# alle
zen_Tief_data_search = all_data_search.loc[
        ((all_data_search["colorimages"] == 1) 
        | (all_data_search["colorimages"] == 0) )&
        ((all_data_search["masktype"] == 2) ) &
        ((all_data_search["maskregion"]  == 2) )
        ,:] 

print("Keine/Mit Farben zentraler Tiefpassfilter", 
      zen_Tief_data_search.shape[0])

# mit Farben
zen_Tief_data_color_search = all_data_search.loc[
        ((all_data_search["colorimages"] == 1)  )&
        ((all_data_search["masktype"] == 1) ) &
        ((all_data_search["maskregion"]  == 2 ) )
        ,:] 

print("Mit Farben zentraler Tiefpassfilter", 
      zen_Tief_data_color_search.shape[0])

# Ohne Farben
zen_Tief_data_grey_search = all_data_search.loc[ 
        ((all_data_search["colorimages"] == 0)  )&
        ((all_data_search["masktype"] == 1) ) &
        ((all_data_search["maskregion"]  == 2 ) )
        ,:] 

print("Ohne Farben zentraler Tiefpassfilter", 
      zen_Tief_data_grey_search.shape[0])



#        masktype == 2 & maskregion == 2: zentraler Hochpassfilter
# alle
zen_Hoch_data_search = all_data_search.loc[ 
        ((all_data_search["colorimages"] == 1)
        | (all_data_search["colorimages"] == 0) )&
        ((all_data_search["masktype"] == 2) ) &
        ((all_data_search["maskregion"]  == 2) )
        ,:] 

print("Keine/Mit Farben zentraler Hochpassfilter", 
      zen_Hoch_data_search.shape[0])

# mit Farben
zen_Hoch_data_color_search = all_data_search.loc[
        ((all_data_search["colorimages"] == 1)  )&
        ((all_data_search["masktype"] == 2) ) &
        ((all_data_search["maskregion"]  == 2 ) )
        ,:] 

print("Mit Farben zentraler Hochpassfilter", 
      zen_Hoch_data_color_search.shape[0])

# Ohne Farben
zen_Hoch_data_grey_search = all_data_search.loc[ 
        ((all_data_search["colorimages"] == 0)  )&
        ((all_data_search["masktype"] == 2) ) &
        ((all_data_search["maskregion"]  == 2 ) )
        ,:] 

print("Ohne Farben zentraler Hochpassfilter", 
      zen_Hoch_data_grey_search.shape[0])


print("---------------------------------------------------------")
print("All Data Memory", all_data.shape[0]) 
print("Mit Farben Kontrollbedingung Memory",
      controll_data_color.shape[0]) 
print("Keine Farben Kontrollbedingung Memory", 
      controll_data_grey.shape[0])
print("Mit Farben peripherer Tiefpassfilter Memory", 
      per_Tief_data_color.shape[0])
print("Ohne Farben peripherer Tiefpassfilter Memory", 
      per_Tief_data_grey.shape[0])
print("Mit Farben peripherer Hochpassfilter Memory", 
      per_Hoch_data_color.shape[0])
print("Ohne Farben peripherer Hochpassfilter Memory", 
      per_Hoch_data_grey.shape[0])


print("Mit Farben zentraler Tiefpassfilter Memory", 
      zen_Tief_data_color.shape[0])
print("Ohne Farben zentraler Tiefpassfilter Memory", 
      zen_Tief_data_grey.shape[0])
print("Mit Farben zentraler Hochpassfilter Memory", 
      zen_Hoch_data_color.shape[0])
print("Ohne Farben zentraler Hochpassfilter Memory", 
      zen_Hoch_data_grey.shape[0])
print("---------------------------------------------------------------------------------------------")
print("All Data Search", all_data_search.shape[0]) 
print("Mit Farben Kontrollbedingung Search", 
      controll_data_color_search.shape[0]) 
print("Keine Farben Kontrollbedingung Search", 
      controll_data_grey_search.shape[0])
print("Mit Farben peripherer Tiefpassfilter Ssearch", 
      per_Tief_data_color_search.shape[0])
print("Ohne Farben peripherer Tiefpassfilter Ssearch", 
      per_Tief_data_grey_search.shape[0])
print("Mit Farben peripherer Hochpassfilter Ssearch", 
      per_Hoch_data_color_search.shape[0])
print("Ohne Farben peripherer Hochpassfilter Ssearch", 
      per_Hoch_data_grey_search.shape[0])


print("Mit Farben zentraler Tiefpassfilter search", 
      zen_Tief_data_color_search.shape[0])
print("Ohne Farben zentraler Tiefpassfilter search",
      zen_Tief_data_grey_search.shape[0])
print("Mit Farben zentraler Hochpassfilter search", 
      zen_Hoch_data_color_search.shape[0])



