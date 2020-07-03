#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:12:14 2019

@author: mendel
"""

from glob import glob
from os import rename
import re




for fname in glob('dataset/Set_01/test/masks/*'):
  rename(fname, re.search("^(.*?)_",os.path.basename(fname)).group(1) +".jpeg")
  
  
  
  
fname  




