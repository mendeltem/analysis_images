#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 11:25:05 2018

@author: DD
"""
import numpy as np
import collections

def analyze_it(iterable_nparrays,
               max_depth = 5,
               max_length = 11,
               _depth = 0,
               _index = 0):
    """Prints a readable short analysis considering type and shape
    of an iterable of/or numpy array(s) (and other types which will
    not be analyzed further).

    Arguments:
        iterable_nparrays: An object that is either a numpy array or
        something iterable with maybe numpy arrays in it.

        max_depth: The maximum depth the analysis will look into.

        max_length: The maximum number of items the analysis will show
        per depth level.

        _depth, _index: Should be untouched and are only for recursive
        use of this function to save the current depth and index.

    Returns:
        No return.
    """
    x = iterable_nparrays
    indentation = ''.join([' ' for i in range(0,_depth)])
    if isinstance(x,np.ndarray):
        print(indentation + 'depth {} index {}          Shape {}'.format(
                _depth, _index, x.shape))
    else:
        print(indentation + 'depth {} index {}          Type {}'.format(
                _depth, _index, type(x)))
    if _depth < max_depth:
        if (isinstance(x, collections.Iterable)
                and not isinstance(x, (str, np.ndarray))):
            for n,element in enumerate(x):
                if n < max_length:
                    analyze_it(element,
                               _depth = _depth + 1,
                               _index =  n,
                               max_length = max_length,
                               max_depth = max_depth)