#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:49:13 2019

@author: mendel
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length',
                             'petal width','target'])
features = ['sepal length', 'sepal width', 'petal length', 'petal width']


# Separating out the features
x = df.loc[:, features].values

# Separating out the target
y = df.loc[:,['target']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)



pca = PCA()

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents[:,:2]
             , columns = ['principal component 1', 'principal component 2'])


finalDf = pd.concat([principalDf, df[['target']]], axis = 1)



per_var = np.round(pca.explained_variance_ratio_ * 100, decimals = 1)


labels = ['PC'+ str(x) for x in range(1, len(per_var)+1)]


plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label =labels)

plt.xlabel('Percentage of Explained Variance')
plt.ylabel('Principle Components')
plt.title("Scree Plot")
plt.show()


print("explained PCA : ",len(pca.explained_variance_))
