#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 12:05:02 2019

@author: DD, JL
"""
import subprocess
from tensorflow.python.client import device_lib

def get_available_gpus():
  """Return list of devices with GPU in device type."""
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_nb_gpus_tf():
  """Return number of GPU devices."""
  n_gpus = get_available_gpus()
  return len(n_gpus)


# tf version allocates memory, which takes time. 
# jl therefore added a subprocess-based version, which however assume that the output of that utility doesn't change

def get_nb_gpus():
  """Return number of GPU devices."""
  n_gpus = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
  return len(n_gpus)