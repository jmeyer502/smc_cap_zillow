
# coding: utf-8

# In[1]:


from boto.s3.connection import S3Connection
import os
import json
import boto.s3
import sys
import datetime
import seaborn as sns
from boto.s3.key import Key
from pprint import pprint
import pandas as pd
import urllib
import csv
import io
import requests
import time
import json
import datetime
from pprint import pprint
import scipy
import numpy as np
import matplotlib.pyplot as plt
color = sns.color_palette()
import math

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import neighbors
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


# In[2]:


rawdata= pd.read_csv("C:/Users/myang/Desktop/Zillow-Data-Analysis-master/Data/train_2017.csv") 
rawdata.shape


# In[3]:


prop_df = pd.read_csv("C:/Users/myang/Desktop/Zillow-Data-Analysis-master/Data/properties_2017.csv")
prop_df.shape


# In[4]:


for c, dtype in zip(prop_df.columns, prop_df.dtypes):
    if dtype == np.float64:
        prop_df[c] = prop_df[c].astype(np.float32)

df_train = rawdata.merge(prop_df, how='left', on='parcelid')
#del prop_df, rawdataspecificrows
df_train.head(2)
#df_train.shape


# trans_mon = month(transactiondate);

# In[5]:


df_train['transaction_mon'] = pd.DatetimeIndex(df_train['transactiondate']).month
df_train.head()


# abs_logerror = abs(logerror);

# In[6]:


df_train['abs_logerror'] = abs(df_train['logerror'])
df_train.head()


# output out=PctlOut P99=P99;

# In[7]:


#99th percentile

max_err = np.percentile(df_train['abs_logerror'], 99)
print(max_err)


# In[8]:


#drop assessment year

df_train = df_train.drop(['assessmentyear'], axis=1)


# Duplicate dataframe to make 0 = good estimate and 1 = bad estimate

# In[9]:


df_copy = df_train.copy()
df_copy.head()


# Creating bad estimate

# In[10]:


df_train['wt'] = df_train['abs_logerror'] / max_err
df_train['weight'] = np.where(df_train['wt'] > 1, 1, df_train['wt'])
df_train['pf'] = 1
df_train.head()


# In[11]:


#drop wt col

df_train.drop('wt', axis=1, inplace=True)
df_train.head()


# Creating good estimate

# In[12]:


df_copy['wt'] = df_copy['abs_logerror'] / max_err
df_copy['wt2'] = np.where(df_copy['wt'] > 1, 1, df_copy['wt'])
df_copy['weight'] = 1 - df_copy['wt2']
df_copy['pf'] = 0
df_copy.head()


# In[13]:


#drop wt, wt2 col

df_copy.drop(['wt', 'wt2'], axis=1, inplace=True)
df_copy.head()


# In[14]:


df_train.shape


# In[15]:


frames = [df_train, df_copy]

data_wt = pd.concat(frames)


# In[16]:


data_wt.shape


# keep only numerical cols

# In[17]:


data_wt_num = data_wt.select_dtypes([np.number])


# In[18]:


data_wt_num.head()


# In[19]:


data_wt_num.shape


# Let's remove fields that have 97% missing values. 97% appears quite subjective, but if you look at https://www.kaggle.com/nikunjm88/creating-additional-features, the variables with more than 97% missing values do not appear to be that important anyways

# In[20]:


missingvalues_prop = (data_wt_num.isnull().sum()/len(data_wt_num)).reset_index()
missingvalues_prop.columns = ['field','proportion']
missingvalues_prop = missingvalues_prop.sort_values(by = 'proportion', ascending = False)
print(missingvalues_prop)
missingvaluescols = missingvalues_prop[missingvalues_prop['proportion'] > 0.97].field.tolist()
data_wt_num = data_wt_num.drop(missingvaluescols, axis=1)


# In[21]:


data_wt_num.shape


# In[22]:


data_wt_copy = data_wt_num.copy()
data_wt_copy = data_wt_copy.fillna(-999999)
data_wt_copy.head()


# In[23]:


#data_wt_num = data_wt_num.drop(['ac_wt'], axis=1)


# In[24]:


data_wt_num['ac_bin'] = pd.cut(data_wt_copy['airconditioningtypeid'], 10, labels=False)
data_wt_num.head()


# In[25]:


data_wt_num['bath_bin'] = pd.cut(data_wt_copy['bathroomcnt'], 10, labels=False)
data_wt_num.head()


# In[26]:


data_wt_num['bed_bin'] = pd.cut(data_wt_copy['bedroomcnt'], 10, labels=False)
data_wt_num.head()


# In[27]:


data_wt_num['bqual_bin'] = pd.cut(data_wt_copy['buildingqualitytypeid'], 10, labels=False)
data_wt_num.head()


# In[28]:


data_wt_num['calcbath_bin'] = pd.cut(data_wt_copy['calculatedbathnbr'], 10, labels=False)
data_wt_num.head()


# In[29]:


# 92% missing data

data_wt_num['ffloor1sqft_bin'] = pd.cut(data_wt_copy['finishedfloor1squarefeet'], 10, labels=False)
data_wt_num.head()


# In[30]:


# use qcut for less missing col?

data_wt_num['calcfinsqft_bin'] = pd.qcut(data_wt_copy['calculatedfinishedsquarefeet'], 10, labels=False)
data_wt_num.head()

# pd.qcut(data_wt_copy['calculatedfinishedsquarefeet'], 10, labels=False)


# In[31]:


data_wt_num['finsqft12_bin'] = pd.qcut(data_wt_copy['finishedsquarefeet12'], 10, labels=False)
data_wt_num.head()

# pd.qcut(data_wt_copy['finishedsquarefeet12'], 10, labels=False)


# In[32]:


data_wt_num['finsqft15_bin'] = pd.cut(data_wt_copy['finishedsquarefeet15'], 10, labels=False)
data_wt_num.head()


# In[33]:


data_wt_num['finsqft50_bin'] = pd.cut(data_wt_copy['finishedsquarefeet50'], 10, labels=False)
data_wt_num.head()


# In[34]:


data_wt_num['fips_bin'] = pd.cut(data_wt_copy['fips'], 10, labels=False)
data_wt_num.head()


# In[35]:


data_wt_num['fullbathcnt_bin'] = pd.cut(data_wt_copy['fullbathcnt'], 10, labels=False)
data_wt_num.head()


# In[36]:


data_wt_num['garagecarcnt_bin'] = pd.cut(data_wt_copy['garagecarcnt'], 10, labels=False)
data_wt_num.head()


# In[37]:


data_wt_num['garagetotalsqft_bin'] = pd.cut(data_wt_copy['garagetotalsqft'], 10, labels=False)
data_wt_num.head()


# In[38]:


data_wt_num['heatsystypeid_bin'] = pd.cut(data_wt_copy['heatingorsystemtypeid'], 10, labels=False)
data_wt_num.head()


# In[39]:


data_wt_num['latitude_bin'] = pd.qcut(data_wt_copy['latitude'], 10, labels=False)
data_wt_num.head()

# pd.qcut(data_wt_copy['latitude'], 10, labels=False)


# In[40]:


data_wt_num['longitude_bin'] = pd.qcut(data_wt_copy['longitude'], 10, labels=False)
data_wt_num.head()

# pd.qcut(data_wt_copy['longitude'], 10, labels=False)


# In[41]:


data_wt_num['lotsizesqft_bin'] = pd.cut(data_wt_copy['lotsizesquarefeet'], 10, labels=False)
data_wt_num.head()


# In[42]:


data_wt_num['poolcnt_bin'] = pd.cut(data_wt_copy['poolcnt'], 10, labels=False)
data_wt_num.head()


# In[43]:


data_wt_num['pooltypeid7_bin'] = pd.cut(data_wt_copy['pooltypeid7'], 10, labels=False)
data_wt_num.head()


# In[44]:


data_wt_num['proplandusetypeid_bin'] = pd.cut(data_wt_copy['propertylandusetypeid'], 10, labels=False)
data_wt_num.head()


# In[45]:


data_wt_num['rawcensus_bin'] = pd.qcut(data_wt_copy['rawcensustractandblock'], 10, labels=False)
data_wt_num.head()

# pd.qcut(data_wt_copy['rawcensustractandblock'], 10, labels=False)


# In[46]:


data_wt_num['regionidcity_bin'] = pd.cut(data_wt_copy['regionidcity'], 10, labels=False)
data_wt_num.head()


# In[47]:


data_wt_num['regionidcounty_bin'] = pd.cut(data_wt_copy['regionidcounty'], 10, labels=False)
data_wt_num.head()


# In[48]:


data_wt_num['regionidneigh_bin'] = pd.cut(data_wt_copy['regionidneighborhood'], 10, labels=False)
data_wt_num.head()


# In[49]:


data_wt_num['regionidzip_bin'] = pd.qcut(data_wt_copy['regionidzip'], 10, labels=False)
data_wt_num.head()

# pd.qcut(data_wt_copy['regionidzip'], 10, labels=False)


# In[50]:


data_wt_num['roomcnt_bin'] = pd.cut(data_wt_copy['roomcnt'], 10, labels=False)
data_wt_num.head()


# In[51]:


data_wt_num['threeqtbath_bin'] = pd.cut(data_wt_copy['threequarterbathnbr'], 10, labels=False)
data_wt_num.head()


# In[52]:


data_wt_num['unitcnt_bin'] = pd.cut(data_wt_copy['unitcnt'], 10, labels=False)
data_wt_num.head()


# In[53]:


data_wt_num['yearbuilt_bin'] = pd.qcut(data_wt_copy['yearbuilt'], 10, labels=False)
data_wt_num.head()


# In[54]:


data_wt_num['numstories_bin'] = pd.cut(data_wt_copy['numberofstories'], 10, labels=False)
data_wt_num.head()


# In[55]:


data_wt_num['staxvaldolcnt_bin'] = pd.qcut(data_wt_copy['structuretaxvaluedollarcnt'], 10, labels=False)
data_wt_num.head()


# In[56]:


data_wt_num['taxvaldolcnt_bin'] = pd.qcut(data_wt_copy['taxvaluedollarcnt'], 10, labels=False)
data_wt_num.head()


# In[57]:


data_wt_num['ltaxvaldolcnt_bin'] = pd.qcut(data_wt_copy['landtaxvaluedollarcnt'], 10, labels=False)
data_wt_num.head()


# In[58]:


data_wt_num['taxamt_bin'] = pd.qcut(data_wt_copy['taxamount'], 10, labels=False)
data_wt_num.head()


# In[59]:


data_wt_num['censustract_bin'] = pd.qcut(data_wt_copy['censustractandblock'], 10, labels=False)
data_wt_num.head()


# In[60]:


data_wt_num.to_csv("Merged_bin_2017.csv", index=False, encoding='utf8')

