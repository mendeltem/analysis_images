#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 18:42:10 2019

@author: pandoora
"""
import numpy as np
import pandas as pd

def rounding(digits,threshold = 0.50):
    """rounds a list of float digits with ad threshhold"""
    if type(digits) == list  or  type(digits) == np.ndarray:
        return np.array(list(map(rounding, digits)))
    if type(digits)== np.float64 or type(digits)== np.float32:
        k = digits % 1
        f = digits - k

        if k >= threshold:
            return (f + 1)
        else:
            return f
    else:
        raise ValueError("Wrong Type")
        
def binary_enc(dataframe, colname):
    """cast a string into a binary int variable
    
    Arguments:
        dataframe: DataFrame to change
        colname:   String Column that has to change
        
    Return:
        the changed DataFrame
    
    """
    temp_df = dataframe.copy(deep=True)

    var_1, var_2 = temp_df.loc[:,colname].unique()
    #print(var_1, ": 1")
    #print(var_2, ": 0")

    mapper = {
        var_1: 1 ,
        var_2: 0 
    }   
    #print(mapper)
    
    temp_df[colname] = temp_df[colname].replace(mapper)
    return temp_df, mapper


def binary_encoding(dataframe, colnames):
    """cast a string into a binary int variable
    
    Arguments:
        dataframe: DataFrame to change
        colname:   String Column that has to change
        
    Return:
        the changed DataFrame
    
    """
    temp_df = dataframe.copy(deep=True)
    
    maps = {   }

    if type(colnames) == list:
        for i, col in enumerate(colnames):
            temp_df, dictionary = binary_enc(temp_df, col)
            #maps.update( {colnames : dictionary} )
            
            maps.update( {col : dictionary} )
    else:
        temp_df, dictionary = binary_enc(temp_df, colnames)
            
    return temp_df,maps
        
def zscore_normalisation(dataframe, colname):
    """zscore transformation 
    
    Arguments:
        dataframe: DataFrame to change
    Return:
        the changed DataFrame        
    """
    temp_df = dataframe.copy(deep=True)
    temp_df[colname] = (temp_df[colname] - temp_df[colname].mean()) / temp_df[colname].std()
    
    return temp_df


def split_dataframe(dataframe, colname):
    """ splitting data frame in to two dataframes
    
    Arguments:
        dataframe: DataFrame to change
        colname:   the selected column would be splitt and returned in a different dataframe
        
    Return:
        dataframe without the given colname
        dataframe with only the colname
    """
    temp_df = dataframe.copy(deep=True)
    
    ordinal_column_df = temp_df.loc[:, colname]
    not_ordinal_df = temp_df.drop(columns=colname)
    return ordinal_column_df, not_ordinal_df