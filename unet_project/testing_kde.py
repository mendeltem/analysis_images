#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:39:36 2019

@author: mendel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:31:26 2019

@author: MUT,DD

Create the training and validation datasets with kernel density estimation 
masks from the eye movement experiment fixations. 
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import seaborn as sns
import re 
import imageio
from skimage.transform import resize
import json
from joblib import Parallel, delayed
import warnings


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors.kde import KernelDensity


warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)

#from modules.library import get_img_paths
with open("input_file.json") as data_file:
    config = json.load(data_file)
#Fixed Value for the Plot
#radnom seed
seed = 0
random.seed(seed)
np.random.seed(seed=seed)
"""Parameter for the process"""
#widh and height
height = 768
width = 1024

#experiment data tables
MEMORY                  = config["MEMORY"]
SEARCH                  = config["SEARCH"]

#input images paths for different imagetypes
MEMORY_COLOR            = config["MEMORY_COLOR"]
MEMORY_GRAYSCALE        = config["MEMORY_GRAYSCALE"]
SEARCH_COLOR            = config["SEARCH_COLOR"]
SEARCH_GRAYSCALE        = config["SEARCH_GRAYSCALE"]

#seed for random 
SEED  = 1
#bandwith parameter for the KDE Masks / Variance of the Gaussian Density

y = np.random.randint(10, 50, size=(1, 200))[0]
x = np.random.randint(10, 50, size=(1, 200))[0]

all_data = pd.read_table(MEMORY) 

all_data.columns

#Code for shape of kernel to fit with. Bivariate KDE can only use gaussian kernel.
#KERNEL : {‘gau’ | ‘cos’ | ‘biw’ | ‘epa’ | ‘tri’ | ‘triw’ }, optional
eye_movements = all_data[( all_data["fixinvalid"] != 1)  
 & (all_data["sacinvalid"] != 1) 
 & (all_data["imageid"] == 1)
 & (all_data["masktype"] == 0)
 & (all_data["maskregion"] == 0)
 
 ].loc[:, ["imageid","fixposx","fixposy"]]

x = list(eye_movements.iloc[:100,0])
y = list(eye_movements.iloc[:100,1])

x = np.array(eye_movements.loc[:,"fixposx"])
y = np.array(eye_movements.loc[:,"fixposy"])

from scipy import stats
kde = stats.gaussian_kde(x)

kde.set_bandwidth(bw_method='scott')
kde.set_bandwidth(bw_method=kde.factor / 3.)
samples = np.random.uniform(0,1,size=(50,2)) 



#kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit([x,y])
#
#kde.
#
#stats.gaussian_kde(x)

#
#plt.contour( [x,y], 
#            levels=[-9998], colors="k",
#            linestyles="solid")

#plt.show()
#
#kde.score_samples(x,y) 
#
#fig, ax = plt.subplots()
#
#ax.plot(x, y, 'k.', markersize=2)
#
#plt.show()

#
#kde2D(x,y,0.3)


#sns.kdeplot(  y[:100],
#              x[:100], 
#              kernel = "gau",
#              bw= 0.10,
#              cmap = plt.cm.gray,
#              shade=True, 
#              n_levels = 100,
#              legend= False,
#              cumulative= False
#              )
#
#plt.show()

def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs): 
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins, 
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)
  
  
def kde2D_plot(x,y, bw,size):  
    
    xx, yy, zz = kde2D(x, y, bw)
    
    fig = plt.pcolormesh(xx, yy, zz)

    plt.close()
   
    fig.axes.get_yaxis().set_visible(False)
    fig.axes.get_xaxis().set_visible(False)
    
    fig.axes.invert_yaxis()
    figure = fig.get_figure()
    figure.tight_layout(pad=0)
    figure.canvas.draw()
    mask = np.fromstring(figure.canvas.tostring_rgb(),dtype=np.uint8,sep='')
    mask = mask.reshape( figure.canvas.get_width_height()[::-1] + (3,))
    
    mask = resize(mask,size,preserve_range = True) 
   
    return mask
  
  
mask = kde2D_plot(x,y, 30, (height, width))


save_image(mask[:,:,0],"test_hkde0.jpg")
save_image(mask[:,:,1],"test_hkde1.jpg")
save_image(mask[:,:,2],"test_hkde2.jpg")
save_image(mask,"test_hkdeall.jpg")


