#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 12:41:55 2018

@author: mendel
"""

import keras
import pandas as pd
import numpy as np



df1 = pd.DataFrame()
df2 = pd.DataFrame()

df1 = df1.assign(e=pd.Series(np.random.randn(10)).values)




df2.concat(df1)



series = pd.Series()




