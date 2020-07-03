#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:26:50 2020

@author: mendel
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import timeout_decorator
tf.enable_eager_execution()

import os
import seaborn as sns
import matplotlib.pyplot as plt

from keras.utils import Progbar
from keras import Sequential
from keras.layers import Dense
from keras.utils.generic_utils import to_list
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.utils import shuffle
import datetime
import re
import random
random.seed = 7

DEBUG = 0

from keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer





import numpy as np
import pandas as pd
import tensorflow as tf
import timeout_decorator
tf.enable_eager_execution()

import os
import seaborn as sns
import matplotlib.pyplot as plt

from keras.utils import Progbar
from keras import Sequential
from keras.layers import Dense
from keras.utils.generic_utils import to_list
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.utils import shuffle
import datetime
import re
import random
random.seed = 7

DEBUG = 0

from keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


from modules.dataloader_featureklassification import Data_Loader



memory_color_path = '/mnt/data/DATA/dataset/FEATURES/saved/exception/memory/color'


layer_filter = "^1_|^2_|^3_|^4_|^5_|^6_|^7_|^8_|^9_|^10_|^11_"

layer_filter = "^1_"

filter1 ="not_filtered"
filter2 =""
batch_size = 300
epochs = 200


loader = Data_Loader(memory_color_path,
                     layer_filter,
                     filter1,
                     filter2,
                     batch_size,
                     train_test_split_ratio = 0.2,
                     showpath = 1,
                     label_type = 0
                     )

len(loader.testpaths)
len(loader.trainpaths)





train_data = pd.DataFrame()


path = loader.trainpaths[1]


print("paths is loading:",path)


part_data = pd.read_feather(path)


















dataframe  = loader.load_train_dataframe()



X, y, labels = next(dataframe)






