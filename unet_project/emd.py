#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:47:01 2019

@author: mendel
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def img_to_sig(arr):
    """Convert a 2D array to a signature for cv2.EMD"""
    
    # cv2.EMD requires single-precision, floating-point input
    sig = np.empty((arr.size, 3), dtype=np.float32)
    count = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            sig[count] = np.array([arr[i,j], i, j])
            count += 1
    return sig
  
  
arr1 = np.array([[2, 0, 0],
                 [2, 0, 0],
                 [0, 0, 2]])

arr2 = np.array([[0, 1, 1],
                 [2, 0, 0],
                 [0, 2, 0]])

sig1 = img_to_sig(arr1)
sig2 = img_to_sig(arr2)

print(sig1)
print(sig2)


dist, _, flow = cv2.EMD(sig1, sig2, cv2.DIST_L2)

print(flow)
