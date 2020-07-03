#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:16:50 2019

@author: mendel
"""
import pandas as pd
import os
# Import plotting modules
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#set the working directory
ML_PATH ="/home/mendel/Documents/work/mendel_exchange"

os.chdir(ML_PATH)

from sklearn.ensemble import RandomForestRegressor

from modules.library import get_csv_paths



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



csv_directory = os.getcwd()+ "/saved/color_high-pass/csv"

all_csv_directorys = get_csv_paths(csv_directory)

feature_data =  pd.read_csv(all_csv_directorys[0])




feature_data.head()



target = feature_data["sacdur"].values

target = np.array(target)


X_df      = feature_data.drop(["sacdur"], axis=1)

columns = X_df.columns

X = X_df.values

rfr = RandomForestRegressor(n_estimators = 400)



small_X = X[:,100:300] 


rfr.fit(small_X ,target)


y_predict = rfr.predict(small_X)


accuracy_score(target,y_predict )





rfr.feature_importances_






## Generate a scatter plot of 'weight' and 'mpg' using red circles
#plt.scatter(auto['weight'], auto['mpg'], label='data', color='red', marker='o')
#
## Plot in blue a linear regression of order 1 between 'weight' and 'mpg'
#sns.regplot(x='weight', y='mpg', data=auto, label="order 1", color="blue", order=1, scatter=None)
#
## Plot in green a linear regression of order 2 between 'weight' and 'mpg'
#sns.regplot(x="weight", y ="mpg", data= auto, label="order 2", color="green", order=2, scatter= None)
#
## Add a legend and display the plot
#plt.legend(loc="upper right")
#plt.show()
#

## Generate a green residual plot of the regression between 'hp' and 'mpg'
#sns.residplot(x='hp', y='mpg', data=auto, color='green')
#
## Display the plot
#plt.show()
#
#
# Plot a linear regression between 'weight' and 'hp', with a hue of 'origin' and palette of 'Set1'
sns.lmplot(x="weight", y = "hp", hue="origin",data =auto, palette="Set1")

# Display the plot
plt.show()



