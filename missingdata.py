# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 21:05:37 2018

@author: Deepika Keswarap
"""

# import libraries

import pandas as pd
from pandas import set_option
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn import model_selection
import scipy as sy
from sklearn.preprocessing import OneHotEncoder
from sklearn import neighbors

# function to read files
def readfiles():
    # read properties2016 file
    prop_2016 = pd.read_csv("C:/Users/Deepika Keswarap/Documents/projecta/all/properties_2016.csv", low_memory=False)
    # read properties2017 file
    prop_2017 = pd.read_csv("C:/Users/Deepika Keswarap/Documents/projecta/all/properties_2017.csv", low_memory=False)
    # read train2016 file 
    train_2016 = pd.read_csv("C:/Users/Deepika Keswarap/Documents/projecta/all//train_2016_v2.csv",
                             parse_dates=["transactiondate"])
    # read train2017 file 
    train_2017 = pd.read_csv("C:/Users/Deepika Keswarap/Documents/projecta/all//train_2017.csv", 
                             parse_dates=["transactiondate"])
    return prop_2016, prop_2017, train_2016, train_2017

# function to merge properties with train files(2016,2017)
def mergefiles():
    prop_2016_error = pd.merge(prop_2016,train_2016,on='parcelid',how = 'outer', left_index = True, right_index=True)
    prop_2017_error = pd.merge(prop_2017,train_2017,on='parcelid',how = 'outer', left_index = True, right_index=True)
    return prop_2016_error, prop_2017_error


# barchart of percent of data missing 
def percentmissingplot():
   print("Percent of data missing in 2016 data")
   print("*****************************************************************")
   missing_df = prop_2016_error.isnull().sum(axis=0).reset_index()
   missing_df.columns = ['column_name', 'missing_count']
   missing_df = missing_df.loc[missing_df['missing_count']>0]
   missing_df = missing_df.sort_values(by='missing_count')

   ind = np.arange(missing_df.shape[0])
   width = 0.9
   fig, ax = plt.subplots(figsize=(12,18))
   rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
   ax.set_yticks(ind)
   ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
   ax.set_xlabel("Count of missing values")
   ax.set_title("Number of missing values in each column")
   plt.show()

   print("Percent of data missing in 2017 data")
   print("*****************************************************************")
   missing_df = prop_2017_error.isnull().sum(axis=0).reset_index()
   missing_df.columns = ['column_name', 'missing_count']
   missing_df = missing_df.loc[missing_df['missing_count']>0]
   missing_df = missing_df.sort_values(by='missing_count')

   ind = np.arange(missing_df.shape[0])
   width = 0.9
   fig, ax = plt.subplots(figsize=(12,18))
   rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
   ax.set_yticks(ind)
   ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
   ax.set_xlabel("Count of missing values")
   ax.set_title("Number of missing values in each column")
   plt.show()
   
# display percent of data missing 
def percentmissing():
    missing = prop_2016_error.isnull().sum().sort_values(ascending = False)
    missingpercent = missing/prop_2016_error['parcelid'].count()
    vartypes = prop_2016_error.dtypes
    pmissing2016 = pd.concat([vartypes, missing, missingpercent], axis = 1, keys =['var type', 'missing rows', 'percent'], sort=True
                     ).sort_values(by = 'missing rows', ascending = False )
    print("Percent of data missing in 2016 data")
    print("*****************************************************************")
    print(pmissing2016)
    
    missing = prop_2017_error.isnull().sum().sort_values(ascending = False)
    missingpercent = missing/prop_2017_error['parcelid'].count()
    vartypes = prop_2017_error.dtypes
    pmissing2017 = pd.concat([vartypes, missing, missingpercent], axis = 1, keys =['var type', 'missing rows', 'percent'], sort=True
                     ).sort_values(by = 'missing rows', ascending = False )
    print("Percent of data missing in 2017 data")
    print("*****************************************************************")
    print(pmissing2017)
    
    
# call functions
prop_2016, prop_2017, train_2016, train_2017 = readfiles()
prop_2016_error, prop_2017_error = mergefiles()
percentmissing()
percentmissingplot()    