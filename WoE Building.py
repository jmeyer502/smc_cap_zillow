
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


data_wt_num = pd.read_csv("C:/Users/myang/Desktop/Zillow-Data-Analysis-master/Merged_bin_2016.csv")


# log(divide((count(*)-sum(&pf.))*&tot_bad.,sum(&pf.)*&tot_gd.)) as woe format=8.4
# 
#  
# 
# ln ( Dist Good / Dist bad)* 100

# In[3]:


data_wt_num.pivot_table(values='weight', index=['pf'], columns=['ac_bin'], aggfunc='sum', margins=True)


# In[4]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['ac_bin'], aggfunc='sum', margins=True)
print(table)


# In[5]:


table2 = table.div(table.iloc[:,-1], axis=0)
print(table2)


# In[6]:


df = table2.reset_index()
df


# In[7]:


df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)
df


# In[8]:


df.divide(df.iloc[1])


# In[9]:


bin0_woe = math.log(.944378)
bin9_woe = math.log(1.140921)


# In[10]:


data_wt_num.loc[data_wt_num.ac_bin == 0, 'ac_woe'] = bin0_woe
data_wt_num.loc[data_wt_num.ac_bin == 9, 'ac_woe'] = bin9_woe


# In[11]:


data_wt_num.head()


# In[12]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['bath_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)
df.divide(df.iloc[1])


# In[13]:


bin0_woe = math.log(.953305)
bin1_woe = math.log(1.135053)
bin2_woe = math.log(0.737987)
bin3_woe = math.log(0.475583)
bin4_woe = math.log(0.45466)
bin5_woe = math.log(0.205035)
bin7_woe = math.log(0.076551)
bin9_woe = math.log(2.057234)

data_wt_num.loc[data_wt_num.bath_bin == 0, 'bath_woe'] = bin0_woe
data_wt_num.loc[data_wt_num.bath_bin == 1, 'bath_woe'] = bin1_woe
data_wt_num.loc[data_wt_num.bath_bin == 2, 'bath_woe'] = bin2_woe
data_wt_num.loc[data_wt_num.bath_bin == 3, 'bath_woe'] = bin3_woe
data_wt_num.loc[data_wt_num.bath_bin == 4, 'bath_woe'] = bin4_woe
data_wt_num.loc[data_wt_num.bath_bin == 5, 'bath_woe'] = bin5_woe
data_wt_num.loc[data_wt_num.bath_bin == 7, 'bath_woe'] = bin7_woe
data_wt_num.loc[data_wt_num.bath_bin == 9, 'bath_woe'] = bin9_woe

data_wt_num.head()


# In[14]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['bed_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[15]:


bin0_woe = math.log(0.785857)
bin1_woe = math.log(1.035347)
bin2_woe = math.log(1.077903)
bin3_woe = math.log(0.816453)
bin4_woe = math.log(0.466075)
bin5_woe = math.log(0.566464)
bin6_woe = math.log(0.405392)
bin7_woe = math.log(0.469175)
bin8_woe = math.log(0.299984)
bin9_woe = math.log(0.298872)

data_wt_num.loc[data_wt_num.bed_bin == 0, 'bed_woe'] = bin0_woe
data_wt_num.loc[data_wt_num.bed_bin == 1, 'bed_woe'] = bin1_woe
data_wt_num.loc[data_wt_num.bed_bin == 2, 'bed_woe'] = bin2_woe
data_wt_num.loc[data_wt_num.bed_bin == 3, 'bed_woe'] = bin3_woe
data_wt_num.loc[data_wt_num.bed_bin == 4, 'bed_woe'] = bin4_woe
data_wt_num.loc[data_wt_num.bed_bin == 5, 'bed_woe'] = bin5_woe
data_wt_num.loc[data_wt_num.bed_bin == 6, 'bed_woe'] = bin6_woe
data_wt_num.loc[data_wt_num.bed_bin == 7, 'bed_woe'] = bin7_woe
data_wt_num.loc[data_wt_num.bed_bin == 8, 'bed_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.bed_bin == 9, 'bed_woe'] = bin9_woe

data_wt_num.head()


# In[16]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['bqual_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[17]:


bin0_woe = math.log(1.186114)
# bin1_woe = math.log(1.035347)
# bin2_woe = math.log(1.077903)
# bin3_woe = math.log(0.816453)
# bin4_woe = math.log(0.466075)
# bin5_woe = math.log(0.566464)
# bin6_woe = math.log(0.405392)
# bin7_woe = math.log(0.469175)
# bin8_woe = math.log(0.299984)
bin9_woe = math.log(0.915459)

data_wt_num.loc[data_wt_num.bqual_bin == 0, 'bqual_woe'] = bin0_woe
# data_wt_num.loc[data_wt_num.ac_bin == 1, 'bed_woe'] = bin1_woe
# data_wt_num.loc[data_wt_num.ac_bin == 2, 'bed_woe'] = bin2_woe
# data_wt_num.loc[data_wt_num.ac_bin == 3, 'bed_woe'] = bin3_woe
# data_wt_num.loc[data_wt_num.ac_bin == 4, 'bed_woe'] = bin4_woe
# data_wt_num.loc[data_wt_num.ac_bin == 5, 'bed_woe'] = bin5_woe
# data_wt_num.loc[data_wt_num.ac_bin == 6, 'bed_woe'] = bin6_woe
# data_wt_num.loc[data_wt_num.ac_bin == 7, 'bed_woe'] = bin7_woe
# data_wt_num.loc[data_wt_num.ac_bin == 8, 'bed_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.bqual_bin == 9, 'bqual_woe'] = bin9_woe

data_wt_num.head()


# In[18]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['calcbath_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[19]:


bin0_woe = math.log(0.459751)
# bin1_woe = math.log(1.035347)
# bin2_woe = math.log(1.077903)
# bin3_woe = math.log(0.816453)
# bin4_woe = math.log(0.466075)
# bin5_woe = math.log(0.566464)
# bin6_woe = math.log(0.405392)
# bin7_woe = math.log(0.469175)
# bin8_woe = math.log(0.299984)
bin9_woe = math.log(1.014098)

data_wt_num.loc[data_wt_num.calcbath_bin == 0, 'calcbath_woe'] = bin0_woe
# data_wt_num.loc[data_wt_num.ac_bin == 1, 'bed_woe'] = bin1_woe
# data_wt_num.loc[data_wt_num.ac_bin == 2, 'bed_woe'] = bin2_woe
# data_wt_num.loc[data_wt_num.ac_bin == 3, 'bed_woe'] = bin3_woe
# data_wt_num.loc[data_wt_num.ac_bin == 4, 'bed_woe'] = bin4_woe
# data_wt_num.loc[data_wt_num.ac_bin == 5, 'bed_woe'] = bin5_woe
# data_wt_num.loc[data_wt_num.ac_bin == 6, 'bed_woe'] = bin6_woe
# data_wt_num.loc[data_wt_num.ac_bin == 7, 'bed_woe'] = bin7_woe
# data_wt_num.loc[data_wt_num.ac_bin == 8, 'bed_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.calcbath_bin == 9, 'calcbath_woe'] = bin9_woe

data_wt_num.head()


# In[20]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['ffloor1sqft_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[21]:


bin0_woe = math.log(0.98784)
# bin1_woe = math.log(1.035347)
# bin2_woe = math.log(1.077903)
# bin3_woe = math.log(0.816453)
# bin4_woe = math.log(0.466075)
# bin5_woe = math.log(0.566464)
# bin6_woe = math.log(0.405392)
# bin7_woe = math.log(0.469175)
# bin8_woe = math.log(0.299984)
bin9_woe = math.log(1.172807)

data_wt_num.loc[data_wt_num.ffloor1sqft_bin == 0, 'ffloor1sqft_woe'] = bin0_woe
# data_wt_num.loc[data_wt_num.ac_bin == 1, 'bed_woe'] = bin1_woe
# data_wt_num.loc[data_wt_num.ac_bin == 2, 'bed_woe'] = bin2_woe
# data_wt_num.loc[data_wt_num.ac_bin == 3, 'bed_woe'] = bin3_woe
# data_wt_num.loc[data_wt_num.ac_bin == 4, 'bed_woe'] = bin4_woe
# data_wt_num.loc[data_wt_num.ac_bin == 5, 'bed_woe'] = bin5_woe
# data_wt_num.loc[data_wt_num.ac_bin == 6, 'bed_woe'] = bin6_woe
# data_wt_num.loc[data_wt_num.ac_bin == 7, 'bed_woe'] = bin7_woe
# data_wt_num.loc[data_wt_num.ac_bin == 8, 'bed_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.ffloor1sqft_bin == 9, 'ffloor1sqft_woe'] = bin9_woe

data_wt_num.head()


# In[22]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['calcfinsqft_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[23]:


bin0_woe = math.log(0.811139)
bin1_woe = math.log(1.007827)
bin2_woe = math.log(1.03572)
bin3_woe = math.log(1.091762)
bin4_woe = math.log(1.097742)
bin5_woe = math.log(1.126556)
bin6_woe = math.log(1.116875)
bin7_woe = math.log(1.079479)
bin8_woe = math.log(1.016829)
bin9_woe = math.log(0.764087)

data_wt_num.loc[data_wt_num.calcfinsqft_bin == 0, 'calcfinsqft_woe'] = bin0_woe
data_wt_num.loc[data_wt_num.calcfinsqft_bin == 1, 'calcfinsqft_woe'] = bin1_woe
data_wt_num.loc[data_wt_num.calcfinsqft_bin == 2, 'calcfinsqft_woe'] = bin2_woe
data_wt_num.loc[data_wt_num.calcfinsqft_bin == 3, 'calcfinsqft_woe'] = bin3_woe
data_wt_num.loc[data_wt_num.calcfinsqft_bin == 4, 'calcfinsqft_woe'] = bin4_woe
data_wt_num.loc[data_wt_num.calcfinsqft_bin == 5, 'calcfinsqft_woe'] = bin5_woe
data_wt_num.loc[data_wt_num.calcfinsqft_bin == 6, 'calcfinsqft_woe'] = bin6_woe
data_wt_num.loc[data_wt_num.calcfinsqft_bin == 7, 'calcfinsqft_woe'] = bin7_woe
data_wt_num.loc[data_wt_num.calcfinsqft_bin == 8, 'calcfinsqft_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.calcfinsqft_bin == 9, 'calcfinsqft_woe'] = bin9_woe

data_wt_num.head()


# In[24]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['finsqft12_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[25]:


bin0_woe = math.log(0.608295)
bin1_woe = math.log(0.96281)
bin2_woe = math.log(1.046762)
bin3_woe = math.log(1.090676)
bin4_woe = math.log(1.123219)
bin5_woe = math.log(1.166815)
bin6_woe = math.log(1.190333)
bin7_woe = math.log(1.175938)
bin8_woe = math.log(1.11884)
bin9_woe = math.log(0.859576)

data_wt_num.loc[data_wt_num.finsqft12_bin == 0, 'finsqft12_woe'] = bin0_woe
data_wt_num.loc[data_wt_num.finsqft12_bin == 1, 'finsqft12_woe'] = bin1_woe
data_wt_num.loc[data_wt_num.finsqft12_bin == 2, 'finsqft12_woe'] = bin2_woe
data_wt_num.loc[data_wt_num.finsqft12_bin == 3, 'finsqft12_woe'] = bin3_woe
data_wt_num.loc[data_wt_num.finsqft12_bin == 4, 'finsqft12_woe'] = bin4_woe
data_wt_num.loc[data_wt_num.finsqft12_bin == 5, 'finsqft12_woe'] = bin5_woe
data_wt_num.loc[data_wt_num.finsqft12_bin == 6, 'finsqft12_woe'] = bin6_woe
data_wt_num.loc[data_wt_num.finsqft12_bin == 7, 'finsqft12_woe'] = bin7_woe
data_wt_num.loc[data_wt_num.finsqft12_bin == 8, 'finsqft12_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.finsqft12_bin == 9, 'finsqft12_woe'] = bin9_woe

data_wt_num.head()


# In[26]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['finsqft15_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[27]:


bin0_woe = math.log(1.041682)
# bin1_woe = math.log(0.96281)
# bin2_woe = math.log(1.046762)
# bin3_woe = math.log(1.090676)
# bin4_woe = math.log(1.123219)
# bin5_woe = math.log(1.166815)
# bin6_woe = math.log(1.190333)
# bin7_woe = math.log(1.175938)
# bin8_woe = math.log(1.11884)
bin9_woe = math.log(0.479441)

data_wt_num.loc[data_wt_num.finsqft15_bin == 0, 'finsqft15_woe'] = bin0_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 1, 'finsqft12_woe'] = bin1_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 2, 'finsqft12_woe'] = bin2_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 3, 'finsqft12_woe'] = bin3_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 4, 'finsqft12_woe'] = bin4_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 5, 'finsqft12_woe'] = bin5_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 6, 'finsqft12_woe'] = bin6_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 7, 'finsqft12_woe'] = bin7_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 8, 'finsqft12_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.finsqft15_bin == 9, 'finsqft15_woe'] = bin9_woe

data_wt_num.head()


# In[28]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['finsqft50_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[29]:


bin0_woe = math.log(0.98784)
# bin1_woe = math.log(0.96281)
# bin2_woe = math.log(1.046762)
# bin3_woe = math.log(1.090676)
# bin4_woe = math.log(1.123219)
# bin5_woe = math.log(1.166815)
# bin6_woe = math.log(1.190333)
# bin7_woe = math.log(1.175938)
# bin8_woe = math.log(1.11884)
bin9_woe = math.log(1.172807)

data_wt_num.loc[data_wt_num.finsqft50_bin == 0, 'finsqft50_woe'] = bin0_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 1, 'finsqft12_woe'] = bin1_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 2, 'finsqft12_woe'] = bin2_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 3, 'finsqft12_woe'] = bin3_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 4, 'finsqft12_woe'] = bin4_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 5, 'finsqft12_woe'] = bin5_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 6, 'finsqft12_woe'] = bin6_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 7, 'finsqft12_woe'] = bin7_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 8, 'finsqft12_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.finsqft50_bin == 9, 'finsqft50_woe'] = bin9_woe

data_wt_num.head()


# In[30]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['fips_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[31]:


bin0_woe = math.log(0.90315)
# bin1_woe = math.log(0.96281)
bin2_woe = math.log(1.276041)
# bin3_woe = math.log(1.090676)
# bin4_woe = math.log(1.123219)
# bin5_woe = math.log(1.166815)
# bin6_woe = math.log(1.190333)
# bin7_woe = math.log(1.175938)
# bin8_woe = math.log(1.11884)
bin9_woe = math.log(1.122243)

data_wt_num.loc[data_wt_num.fips_bin == 0, 'fips_woe'] = bin0_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 1, 'finsqft12_woe'] = bin1_woe
data_wt_num.loc[data_wt_num.fips_bin == 2, 'fips_woe'] = bin2_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 3, 'finsqft12_woe'] = bin3_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 4, 'finsqft12_woe'] = bin4_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 5, 'finsqft12_woe'] = bin5_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 6, 'finsqft12_woe'] = bin6_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 7, 'finsqft12_woe'] = bin7_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 8, 'finsqft12_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.fips_bin == 9, 'fips_woe'] = bin9_woe

data_wt_num.head()


# In[32]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['fireplacecnt_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[33]:


bin0_woe = math.log(0.975288)
# bin1_woe = math.log(0.96281)
# bin2_woe = math.log(1.276041)
# bin3_woe = math.log(1.090676)
# bin4_woe = math.log(1.123219)
# bin5_woe = math.log(1.166815)
# bin6_woe = math.log(1.190333)
# bin7_woe = math.log(1.175938)
# bin8_woe = math.log(1.11884)
bin9_woe = math.log(1.262171)

data_wt_num.loc[data_wt_num.fireplacecnt_bin == 0, 'fireplacecnt_woe'] = bin0_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 1, 'finsqft12_woe'] = bin1_woe
# data_wt_num.loc[data_wt_num.fips_bin == 2, 'fips_woe'] = bin2_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 3, 'finsqft12_woe'] = bin3_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 4, 'finsqft12_woe'] = bin4_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 5, 'finsqft12_woe'] = bin5_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 6, 'finsqft12_woe'] = bin6_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 7, 'finsqft12_woe'] = bin7_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 8, 'finsqft12_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.fireplacecnt_bin == 9, 'fireplacecnt_woe'] = bin9_woe

data_wt_num.head()


# In[34]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['fullbathcnt_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[35]:


bin0_woe = math.log(0.459751)
# bin1_woe = math.log(0.96281)
# bin2_woe = math.log(1.276041)
# bin3_woe = math.log(1.090676)
# bin4_woe = math.log(1.123219)
# bin5_woe = math.log(1.166815)
# bin6_woe = math.log(1.190333)
# bin7_woe = math.log(1.175938)
# bin8_woe = math.log(1.11884)
bin9_woe = math.log(1.014098)

data_wt_num.loc[data_wt_num.fullbathcnt_bin == 0, 'fullbathcnt_woe'] = bin0_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 1, 'finsqft12_woe'] = bin1_woe
# data_wt_num.loc[data_wt_num.fips_bin == 2, 'fips_woe'] = bin2_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 3, 'finsqft12_woe'] = bin3_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 4, 'finsqft12_woe'] = bin4_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 5, 'finsqft12_woe'] = bin5_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 6, 'finsqft12_woe'] = bin6_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 7, 'finsqft12_woe'] = bin7_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 8, 'finsqft12_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.fullbathcnt_bin == 9, 'fullbathcnt_woe'] = bin9_woe

data_wt_num.head()


# In[36]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['garagecarcnt_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[37]:


bin0_woe = math.log(0.895071)
# bin1_woe = math.log(0.96281)
# bin2_woe = math.log(1.276041)
# bin3_woe = math.log(1.090676)
# bin4_woe = math.log(1.123219)
# bin5_woe = math.log(1.166815)
# bin6_woe = math.log(1.190333)
# bin7_woe = math.log(1.175938)
# bin8_woe = math.log(1.11884)
bin9_woe = math.log(1.295431)

data_wt_num.loc[data_wt_num.garagecarcnt_bin == 0, 'garagecarcnt_woe'] = bin0_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 1, 'finsqft12_woe'] = bin1_woe
# data_wt_num.loc[data_wt_num.fips_bin == 2, 'fips_woe'] = bin2_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 3, 'finsqft12_woe'] = bin3_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 4, 'finsqft12_woe'] = bin4_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 5, 'finsqft12_woe'] = bin5_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 6, 'finsqft12_woe'] = bin6_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 7, 'finsqft12_woe'] = bin7_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 8, 'finsqft12_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.garagecarcnt_bin == 9, 'garagecarcnt_woe'] = bin9_woe

data_wt_num.head()


# In[38]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['garagetotalsqft_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[39]:


bin0_woe = math.log(0.895071)
# bin1_woe = math.log(0.96281)
# bin2_woe = math.log(1.276041)
# bin3_woe = math.log(1.090676)
# bin4_woe = math.log(1.123219)
# bin5_woe = math.log(1.166815)
# bin6_woe = math.log(1.190333)
# bin7_woe = math.log(1.175938)
# bin8_woe = math.log(1.11884)
bin9_woe = math.log(1.295431)

data_wt_num.loc[data_wt_num.garagetotalsqft_bin == 0, 'garagetotalsqft_woe'] = bin0_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 1, 'finsqft12_woe'] = bin1_woe
# data_wt_num.loc[data_wt_num.fips_bin == 2, 'fips_woe'] = bin2_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 3, 'finsqft12_woe'] = bin3_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 4, 'finsqft12_woe'] = bin4_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 5, 'finsqft12_woe'] = bin5_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 6, 'finsqft12_woe'] = bin6_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 7, 'finsqft12_woe'] = bin7_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 8, 'finsqft12_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.garagetotalsqft_bin == 9, 'garagetotalsqft_woe'] = bin9_woe

data_wt_num.head()


# In[40]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['heatsystypeid_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[41]:


bin0_woe = math.log(1.04182)
# bin1_woe = math.log(0.96281)
# bin2_woe = math.log(1.276041)
# bin3_woe = math.log(1.090676)
# bin4_woe = math.log(1.123219)
# bin5_woe = math.log(1.166815)
# bin6_woe = math.log(1.190333)
# bin7_woe = math.log(1.175938)
# bin8_woe = math.log(1.11884)
bin9_woe = math.log(0.975954)

data_wt_num.loc[data_wt_num.heatsystypeid_bin == 0, 'heatsystypeid_woe'] = bin0_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 1, 'finsqft12_woe'] = bin1_woe
# data_wt_num.loc[data_wt_num.fips_bin == 2, 'fips_woe'] = bin2_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 3, 'finsqft12_woe'] = bin3_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 4, 'finsqft12_woe'] = bin4_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 5, 'finsqft12_woe'] = bin5_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 6, 'finsqft12_woe'] = bin6_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 7, 'finsqft12_woe'] = bin7_woe
# data_wt_num.loc[data_wt_num.finsqft12_bin == 8, 'finsqft12_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.heatsystypeid_bin == 9, 'heatsystypeid_woe'] = bin9_woe

data_wt_num.head()


# In[42]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['latitude_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[43]:


bin0_woe = math.log(1.145724)
bin1_woe = math.log(1.238128)
bin2_woe = math.log(1.199759)
bin3_woe = math.log(1.113028)
bin4_woe = math.log(0.77322)
bin5_woe = math.log(0.768164)
bin6_woe = math.log(0.871661)
bin7_woe = math.log(0.963885)
bin8_woe = math.log(1.063063)
bin9_woe = math.log(1.113822)

data_wt_num.loc[data_wt_num.latitude_bin == 0, 'latitude_woe'] = bin0_woe
data_wt_num.loc[data_wt_num.latitude_bin == 1, 'latitude_woe'] = bin1_woe
data_wt_num.loc[data_wt_num.latitude_bin == 2, 'latitude_woe'] = bin2_woe
data_wt_num.loc[data_wt_num.latitude_bin == 3, 'latitude_woe'] = bin3_woe
data_wt_num.loc[data_wt_num.latitude_bin == 4, 'latitude_woe'] = bin4_woe
data_wt_num.loc[data_wt_num.latitude_bin == 5, 'latitude_woe'] = bin5_woe
data_wt_num.loc[data_wt_num.latitude_bin == 6, 'latitude_woe'] = bin6_woe
data_wt_num.loc[data_wt_num.latitude_bin == 7, 'latitude_woe'] = bin7_woe
data_wt_num.loc[data_wt_num.latitude_bin == 8, 'latitude_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.latitude_bin == 9, 'latitude_woe'] = bin9_woe

data_wt_num.head()


# In[44]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['longitude_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[45]:


bin0_woe = math.log(1.085999)
bin1_woe = math.log(1.047752)
bin2_woe = math.log(0.851783)
bin3_woe = math.log(0.811902)
bin4_woe = math.log(0.732826)
bin5_woe = math.log(0.95255)
bin6_woe = math.log(1.119131)
bin7_woe = math.log(1.098016)
bin8_woe = math.log(1.226646)
bin9_woe = math.log(1.386212)

data_wt_num.loc[data_wt_num.longitude_bin == 0, 'longitude_woe'] = bin0_woe
data_wt_num.loc[data_wt_num.longitude_bin == 1, 'longitude_woe'] = bin1_woe
data_wt_num.loc[data_wt_num.longitude_bin == 2, 'longitude_woe'] = bin2_woe
data_wt_num.loc[data_wt_num.longitude_bin == 3, 'longitude_woe'] = bin3_woe
data_wt_num.loc[data_wt_num.longitude_bin == 4, 'longitude_woe'] = bin4_woe
data_wt_num.loc[data_wt_num.longitude_bin == 5, 'longitude_woe'] = bin5_woe
data_wt_num.loc[data_wt_num.longitude_bin == 6, 'longitude_woe'] = bin6_woe
data_wt_num.loc[data_wt_num.longitude_bin == 7, 'longitude_woe'] = bin7_woe
data_wt_num.loc[data_wt_num.longitude_bin == 8, 'longitude_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.longitude_bin == 9, 'longitude_woe'] = bin9_woe

data_wt_num.head()


# In[46]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['lotsizesqft_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[47]:


bin0_woe = math.log(1.291853)
bin1_woe = math.log(0.971177)
bin2_woe = math.log(1.220901)
bin3_woe = math.log(0.340023)
bin4_woe = math.log(0.317152)
bin5_woe = math.log(1.679497)
# bin6_woe = math.log(1.119131)
# bin7_woe = math.log(1.098016)
# bin8_woe = math.log(1.226646)
bin9_woe = math.log(0.77095)

data_wt_num.loc[data_wt_num.lotsizesqft_bin == 0, 'lotsizesqft_woe'] = bin0_woe
data_wt_num.loc[data_wt_num.lotsizesqft_bin == 1, 'lotsizesqft_woe'] = bin1_woe
data_wt_num.loc[data_wt_num.lotsizesqft_bin == 2, 'lotsizesqft_woe'] = bin2_woe
data_wt_num.loc[data_wt_num.lotsizesqft_bin == 3, 'lotsizesqft_woe'] = bin3_woe
data_wt_num.loc[data_wt_num.lotsizesqft_bin == 4, 'lotsizesqft_woe'] = bin4_woe
data_wt_num.loc[data_wt_num.lotsizesqft_bin == 5, 'lotsizesqft_woe'] = bin5_woe
# data_wt_num.loc[data_wt_num.longitude_bin == 6, 'longitude_woe'] = bin6_woe
# data_wt_num.loc[data_wt_num.longitude_bin == 7, 'longitude_woe'] = bin7_woe
# data_wt_num.loc[data_wt_num.longitude_bin == 8, 'longitude_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.lotsizesqft_bin == 9, 'lotsizesqft_woe'] = bin9_woe

data_wt_num.head()


# In[48]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['poolcnt_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[49]:


bin0_woe = math.log(0.990498)
# bin1_woe = math.log(0.971177)
# bin2_woe = math.log(1.220901)
# bin3_woe = math.log(0.340023)
# bin4_woe = math.log(0.317152)
# bin5_woe = math.log(1.679497)
# bin6_woe = math.log(1.119131)
# bin7_woe = math.log(1.098016)
# bin8_woe = math.log(1.226646)
bin9_woe = math.log(1.040145)

data_wt_num.loc[data_wt_num.poolcnt_bin == 0, 'poolcnt_woe'] = bin0_woe
# data_wt_num.loc[data_wt_num.lotsizesqft_bin == 1, 'lotsizesqft_woe'] = bin1_woe
# data_wt_num.loc[data_wt_num.lotsizesqft_bin == 2, 'lotsizesqft_woe'] = bin2_woe
# data_wt_num.loc[data_wt_num.lotsizesqft_bin == 3, 'lotsizesqft_woe'] = bin3_woe
# data_wt_num.loc[data_wt_num.lotsizesqft_bin == 4, 'lotsizesqft_woe'] = bin4_woe
# data_wt_num.loc[data_wt_num.lotsizesqft_bin == 5, 'lotsizesqft_woe'] = bin5_woe
# data_wt_num.loc[data_wt_num.longitude_bin == 6, 'longitude_woe'] = bin6_woe
# data_wt_num.loc[data_wt_num.longitude_bin == 7, 'longitude_woe'] = bin7_woe
# data_wt_num.loc[data_wt_num.longitude_bin == 8, 'longitude_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.poolcnt_bin == 9, 'poolcnt_woe'] = bin9_woe

data_wt_num.head()


# In[50]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['pooltypeid7_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[51]:


bin0_woe = math.log(0.99307)
# bin1_woe = math.log(0.971177)
# bin2_woe = math.log(1.220901)
# bin3_woe = math.log(0.340023)
# bin4_woe = math.log(0.317152)
# bin5_woe = math.log(1.679497)
# bin6_woe = math.log(1.119131)
# bin7_woe = math.log(1.098016)
# bin8_woe = math.log(1.226646)
bin9_woe = math.log(1.031601)

data_wt_num.loc[data_wt_num.pooltypeid7_bin == 0, 'pooltypeid7_woe'] = bin0_woe
# data_wt_num.loc[data_wt_num.lotsizesqft_bin == 1, 'lotsizesqft_woe'] = bin1_woe
# data_wt_num.loc[data_wt_num.lotsizesqft_bin == 2, 'lotsizesqft_woe'] = bin2_woe
# data_wt_num.loc[data_wt_num.lotsizesqft_bin == 3, 'lotsizesqft_woe'] = bin3_woe
# data_wt_num.loc[data_wt_num.lotsizesqft_bin == 4, 'lotsizesqft_woe'] = bin4_woe
# data_wt_num.loc[data_wt_num.lotsizesqft_bin == 5, 'lotsizesqft_woe'] = bin5_woe
# data_wt_num.loc[data_wt_num.longitude_bin == 6, 'longitude_woe'] = bin6_woe
# data_wt_num.loc[data_wt_num.longitude_bin == 7, 'longitude_woe'] = bin7_woe
# data_wt_num.loc[data_wt_num.longitude_bin == 8, 'longitude_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.pooltypeid7_bin == 9, 'pooltypeid7_woe'] = bin9_woe

data_wt_num.head()


# In[52]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['proplandusetypeid_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[53]:


bin0_woe = math.log(0.155884)
# bin1_woe = math.log(0.971177)
# bin2_woe = math.log(1.220901)
# bin3_woe = math.log(0.340023)
# bin4_woe = math.log(0.317152)
# bin5_woe = math.log(1.679497)
# bin6_woe = math.log(1.119131)
# bin7_woe = math.log(1.098016)
bin8_woe = math.log(0.483021)
bin9_woe = math.log(1.04594)

data_wt_num.loc[data_wt_num.proplandusetypeid_bin == 0, 'proplandusetypeid_woe'] = bin0_woe
# data_wt_num.loc[data_wt_num.lotsizesqft_bin == 1, 'lotsizesqft_woe'] = bin1_woe
# data_wt_num.loc[data_wt_num.lotsizesqft_bin == 2, 'lotsizesqft_woe'] = bin2_woe
# data_wt_num.loc[data_wt_num.lotsizesqft_bin == 3, 'lotsizesqft_woe'] = bin3_woe
# data_wt_num.loc[data_wt_num.lotsizesqft_bin == 4, 'lotsizesqft_woe'] = bin4_woe
# data_wt_num.loc[data_wt_num.lotsizesqft_bin == 5, 'lotsizesqft_woe'] = bin5_woe
# data_wt_num.loc[data_wt_num.longitude_bin == 6, 'longitude_woe'] = bin6_woe
# data_wt_num.loc[data_wt_num.longitude_bin == 7, 'longitude_woe'] = bin7_woe
data_wt_num.loc[data_wt_num.proplandusetypeid_bin == 8, 'proplandusetypeid_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.proplandusetypeid_bin == 9, 'proplandusetypeid_woe'] = bin9_woe

data_wt_num.head()


# In[54]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['rawcensus_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[55]:


bin0_woe = math.log(0.964668)
bin1_woe = math.log(0.643506)
bin2_woe = math.log(1.026966)
bin3_woe = math.log(0.912237)
bin4_woe = math.log(0.947049)
bin5_woe = math.log(0.86939)
bin6_woe = math.log(1.501193)
bin7_woe = math.log(1.189915)
bin8_woe = math.log(1.156161)
bin9_woe = math.log(1.168805)

data_wt_num.loc[data_wt_num.rawcensus_bin == 0, 'rawcensus_woe'] = bin0_woe
data_wt_num.loc[data_wt_num.rawcensus_bin == 1, 'rawcensus_woe'] = bin1_woe
data_wt_num.loc[data_wt_num.rawcensus_bin == 2, 'rawcensus_woe'] = bin2_woe
data_wt_num.loc[data_wt_num.rawcensus_bin == 3, 'rawcensus_woe'] = bin3_woe
data_wt_num.loc[data_wt_num.rawcensus_bin == 4, 'rawcensus_woe'] = bin4_woe
data_wt_num.loc[data_wt_num.rawcensus_bin == 5, 'rawcensus_woe'] = bin5_woe
data_wt_num.loc[data_wt_num.rawcensus_bin == 6, 'rawcensus_woe'] = bin6_woe
data_wt_num.loc[data_wt_num.rawcensus_bin == 7, 'rawcensus_woe'] = bin7_woe
data_wt_num.loc[data_wt_num.rawcensus_bin == 8, 'rawcensus_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.rawcensus_bin == 9, 'rawcensus_woe'] = bin9_woe

data_wt_num.head()


# In[56]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['regionidcity_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[58]:


bin0_woe = math.log(1.184505)
# bin1_woe = math.log(0.643506)
# bin2_woe = math.log(1.026966)
# bin3_woe = math.log(0.912237)
# bin4_woe = math.log(0.947049)
# bin5_woe = math.log(0.86939)
# bin6_woe = math.log(1.501193)
bin7_woe = math.log(0.999329)
bin8_woe = math.log(0.953636)
bin9_woe = math.log(0.869803)

data_wt_num.loc[data_wt_num.regionidcity_bin == 0, 'regionidcity_woe'] = bin0_woe
# data_wt_num.loc[data_wt_num.regionidcity_bin == 1, 'regionidcity_woe'] = bin1_woe
# data_wt_num.loc[data_wt_num.regionidcity_bin == 2, 'regionidcity_woe'] = bin2_woe
# data_wt_num.loc[data_wt_num.regionidcity_bin == 3, 'regionidcity_woe'] = bin3_woe
# data_wt_num.loc[data_wt_num.regionidcity_bin == 4, 'regionidcity_woe'] = bin4_woe
# data_wt_num.loc[data_wt_num.regionidcity_bin == 5, 'regionidcity_woe'] = bin5_woe
# data_wt_num.loc[data_wt_num.regionidcity_bin == 6, 'regionidcity_woe'] = bin6_woe
data_wt_num.loc[data_wt_num.regionidcity_bin == 7, 'regionidcity_woe'] = bin7_woe
data_wt_num.loc[data_wt_num.regionidcity_bin == 8, 'regionidcity_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.regionidcity_bin == 9, 'regionidcity_woe'] = bin9_woe

data_wt_num.head()


# In[59]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['regionidcounty_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[60]:


bin0_woe = math.log(1.276041)
# bin1_woe = math.log(0.643506)
# bin2_woe = math.log(1.026966)
# bin3_woe = math.log(0.912237)
bin4_woe = math.log(1.122243)
# bin5_woe = math.log(0.86939)
# bin6_woe = math.log(1.501193)
# bin7_woe = math.log(0.999329)
# bin8_woe = math.log(0.953636)
bin9_woe = math.log(0.90315)

data_wt_num.loc[data_wt_num.regionidcounty_bin == 0, 'regionidcounty_woe'] = bin0_woe
# data_wt_num.loc[data_wt_num.regionidcity_bin == 1, 'regionidcity_woe'] = bin1_woe
# data_wt_num.loc[data_wt_num.regionidcity_bin == 2, 'regionidcity_woe'] = bin2_woe
# data_wt_num.loc[data_wt_num.regionidcity_bin == 3, 'regionidcity_woe'] = bin3_woe
data_wt_num.loc[data_wt_num.regionidcounty_bin == 4, 'regionidcounty_woe'] = bin4_woe
# data_wt_num.loc[data_wt_num.regionidcity_bin == 5, 'regionidcity_woe'] = bin5_woe
# data_wt_num.loc[data_wt_num.regionidcity_bin == 6, 'regionidcity_woe'] = bin6_woe
# data_wt_num.loc[data_wt_num.regionidcity_bin == 7, 'regionidcity_woe'] = bin7_woe
# data_wt_num.loc[data_wt_num.regionidcity_bin == 8, 'regionidcity_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.regionidcounty_bin == 9, 'regionidcounty_woe'] = bin9_woe

data_wt_num.head()


# In[61]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['regionidneigh_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[62]:


bin0_woe = math.log(1.068202)
# bin1_woe = math.log(0.643506)
# bin2_woe = math.log(1.026966)
# bin3_woe = math.log(0.912237)
# bin4_woe = math.log(1.122243)
bin5_woe = math.log(0.987979)
bin6_woe = math.log(0.75258)
bin7_woe = math.log(0.907024)
bin8_woe = math.log(1.017867)
bin9_woe = math.log(0.925322)

data_wt_num.loc[data_wt_num.regionidneigh_bin == 0, 'regionidneigh_woe'] = bin0_woe
# data_wt_num.loc[data_wt_num.regionidcity_bin == 1, 'regionidcity_woe'] = bin1_woe
# data_wt_num.loc[data_wt_num.regionidcity_bin == 2, 'regionidcity_woe'] = bin2_woe
# data_wt_num.loc[data_wt_num.regionidcity_bin == 3, 'regionidcity_woe'] = bin3_woe
# data_wt_num.loc[data_wt_num.regionidneigh_bin == 4, 'regionidneigh_woe'] = bin4_woe
data_wt_num.loc[data_wt_num.regionidneigh_bin == 5, 'regionidneigh_woe'] = bin5_woe
data_wt_num.loc[data_wt_num.regionidneigh_bin == 6, 'regionidneigh_woe'] = bin6_woe
data_wt_num.loc[data_wt_num.regionidneigh_bin == 7, 'regionidneigh_woe'] = bin7_woe
data_wt_num.loc[data_wt_num.regionidneigh_bin == 8, 'regionidneigh_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.regionidneigh_bin == 9, 'regionidneigh_woe'] = bin9_woe

data_wt_num.head()


# In[63]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['regionidzip_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[64]:


bin0_woe = math.log(0.650165)
bin1_woe = math.log(0.752836)
bin2_woe = math.log(1.147472)
bin3_woe = math.log(0.966434)
bin4_woe = math.log(1.175751)
bin5_woe = math.log(1.037657)
bin6_woe = math.log(1.136142)
bin7_woe = math.log(1.214425)
bin8_woe = math.log(1.296473)
bin9_woe = math.log(1.01469)

data_wt_num.loc[data_wt_num.regionidzip_bin == 0, 'regionidzip_woe'] = bin0_woe
data_wt_num.loc[data_wt_num.regionidzip_bin == 1, 'regionidzip_woe'] = bin1_woe
data_wt_num.loc[data_wt_num.regionidzip_bin == 2, 'regionidzip_woe'] = bin2_woe
data_wt_num.loc[data_wt_num.regionidzip_bin == 3, 'regionidzip_woe'] = bin3_woe
data_wt_num.loc[data_wt_num.regionidzip_bin == 4, 'regionidzip_woe'] = bin4_woe
data_wt_num.loc[data_wt_num.regionidzip_bin == 5, 'regionidzip_woe'] = bin5_woe
data_wt_num.loc[data_wt_num.regionidzip_bin == 6, 'regionidzip_woe'] = bin6_woe
data_wt_num.loc[data_wt_num.regionidzip_bin == 7, 'regionidzip_woe'] = bin7_woe
data_wt_num.loc[data_wt_num.regionidzip_bin == 8, 'regionidzip_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.regionidzip_bin == 9, 'regionidzip_woe'] = bin9_woe

data_wt_num.head()


# In[65]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['roomcnt_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[66]:


bin0_woe = math.log(0.95909)
bin1_woe = math.log(0.767398)
bin2_woe = math.log(0.971905)
bin3_woe = math.log(1.257854)
bin4_woe = math.log(1.317647)
bin5_woe = math.log(1.090174)
bin6_woe = math.log(0.698365)
bin7_woe = math.log(0.420684)
bin8_woe = math.log(1.006293)
bin9_woe = math.log(4.26848)

data_wt_num.loc[data_wt_num.roomcnt_bin == 0, 'roomcnt_woe'] = bin0_woe
data_wt_num.loc[data_wt_num.roomcnt_bin == 1, 'roomcnt_woe'] = bin1_woe
data_wt_num.loc[data_wt_num.roomcnt_bin == 2, 'roomcnt_woe'] = bin2_woe
data_wt_num.loc[data_wt_num.roomcnt_bin == 3, 'roomcnt_woe'] = bin3_woe
data_wt_num.loc[data_wt_num.roomcnt_bin == 4, 'roomcnt_woe'] = bin4_woe
data_wt_num.loc[data_wt_num.roomcnt_bin == 5, 'roomcnt_woe'] = bin5_woe
data_wt_num.loc[data_wt_num.roomcnt_bin == 6, 'roomcnt_woe'] = bin6_woe
data_wt_num.loc[data_wt_num.roomcnt_bin == 7, 'roomcnt_woe'] = bin7_woe
data_wt_num.loc[data_wt_num.roomcnt_bin == 8, 'roomcnt_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.roomcnt_bin == 9, 'roomcnt_woe'] = bin9_woe

data_wt_num.head()


# In[67]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['threeqtbath_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[68]:


bin0_woe = math.log(0.95909)
# bin1_woe = math.log(0.767398)
# bin2_woe = math.log(0.971905)
# bin3_woe = math.log(1.257854)
# bin4_woe = math.log(1.317647)
# bin5_woe = math.log(1.090174)
# bin6_woe = math.log(0.698365)
# bin7_woe = math.log(0.420684)
# bin8_woe = math.log(1.006293)
bin9_woe = math.log(1.373155)

data_wt_num.loc[data_wt_num.threeqtbath_bin == 0, 'threeqtbath_woe'] = bin0_woe
# data_wt_num.loc[data_wt_num.threeqtbath_bin == 1, 'threeqtbath_woe'] = bin1_woe
# data_wt_num.loc[data_wt_num.threeqtbath_bin == 2, 'threeqtbath_woe'] = bin2_woe
# data_wt_num.loc[data_wt_num.threeqtbath_bin == 3, 'threeqtbath_woe'] = bin3_woe
# data_wt_num.loc[data_wt_num.threeqtbath_bin == 4, 'threeqtbath_woe'] = bin4_woe
# data_wt_num.loc[data_wt_num.threeqtbath_bin == 5, 'threeqtbath_woe'] = bin5_woe
# data_wt_num.loc[data_wt_num.threeqtbath_bin == 6, 'threeqtbath_woe'] = bin6_woe
# data_wt_num.loc[data_wt_num.threeqtbath_bin == 7, 'threeqtbath_woe'] = bin7_woe
# data_wt_num.loc[data_wt_num.threeqtbath_bin == 8, 'threeqtbath_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.threeqtbath_bin == 9, 'threeqtbath_woe'] = bin9_woe

data_wt_num.head()


# In[69]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['unitcnt_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[70]:


bin0_woe = math.log(1.233742)
# bin1_woe = math.log(0.767398)
# bin2_woe = math.log(0.971905)
# bin3_woe = math.log(1.257854)
# bin4_woe = math.log(1.317647)
# bin5_woe = math.log(1.090174)
# bin6_woe = math.log(0.698365)
# bin7_woe = math.log(0.420684)
# bin8_woe = math.log(1.006293)
bin9_woe = math.log(0.903476)

data_wt_num.loc[data_wt_num.unitcnt_bin == 0, 'unitcnt_woe'] = bin0_woe
# data_wt_num.loc[data_wt_num.threeqtbath_bin == 1, 'threeqtbath_woe'] = bin1_woe
# data_wt_num.loc[data_wt_num.threeqtbath_bin == 2, 'threeqtbath_woe'] = bin2_woe
# data_wt_num.loc[data_wt_num.threeqtbath_bin == 3, 'threeqtbath_woe'] = bin3_woe
# data_wt_num.loc[data_wt_num.threeqtbath_bin == 4, 'threeqtbath_woe'] = bin4_woe
# data_wt_num.loc[data_wt_num.threeqtbath_bin == 5, 'threeqtbath_woe'] = bin5_woe
# data_wt_num.loc[data_wt_num.threeqtbath_bin == 6, 'threeqtbath_woe'] = bin6_woe
# data_wt_num.loc[data_wt_num.threeqtbath_bin == 7, 'threeqtbath_woe'] = bin7_woe
# data_wt_num.loc[data_wt_num.threeqtbath_bin == 8, 'threeqtbath_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.unitcnt_bin == 9, 'unitcnt_woe'] = bin9_woe

data_wt_num.head()


# In[71]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['yearbuilt_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[72]:


bin0_woe = math.log(0.571248)
bin1_woe = math.log(0.774187)
bin2_woe = math.log(0.932667)
bin3_woe = math.log(0.969511)
bin4_woe = math.log(1.082327)
bin5_woe = math.log(1.154682)
bin6_woe = math.log(1.166161)
bin7_woe = math.log(1.346666)
bin8_woe = math.log(1.382119)
bin9_woe = math.log(1.317868)

data_wt_num.loc[data_wt_num.yearbuilt_bin == 0, 'yearbuilt_woe'] = bin0_woe
data_wt_num.loc[data_wt_num.yearbuilt_bin == 1, 'yearbuilt_woe'] = bin1_woe
data_wt_num.loc[data_wt_num.yearbuilt_bin == 2, 'yearbuilt_woe'] = bin2_woe
data_wt_num.loc[data_wt_num.yearbuilt_bin == 3, 'yearbuilt_woe'] = bin3_woe
data_wt_num.loc[data_wt_num.yearbuilt_bin == 4, 'yearbuilt_woe'] = bin4_woe
data_wt_num.loc[data_wt_num.yearbuilt_bin == 5, 'yearbuilt_woe'] = bin5_woe
data_wt_num.loc[data_wt_num.yearbuilt_bin == 6, 'yearbuilt_woe'] = bin6_woe
data_wt_num.loc[data_wt_num.yearbuilt_bin == 7, 'yearbuilt_woe'] = bin7_woe
data_wt_num.loc[data_wt_num.yearbuilt_bin == 8, 'yearbuilt_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.yearbuilt_bin == 9, 'yearbuilt_woe'] = bin9_woe

data_wt_num.head()


# In[73]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['numstories_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[74]:


bin0_woe = math.log(0.963901)
# bin1_woe = math.log(0.774187)
# bin2_woe = math.log(0.932667)
# bin3_woe = math.log(0.969511)
# bin4_woe = math.log(1.082327)
# bin5_woe = math.log(1.154682)
# bin6_woe = math.log(1.166161)
# bin7_woe = math.log(1.346666)
# bin8_woe = math.log(1.382119)
bin9_woe = math.log(1.142617)

data_wt_num.loc[data_wt_num.numstories_bin == 0, 'numstories_woe'] = bin0_woe
# data_wt_num.loc[data_wt_num.yearbuilt_bin == 1, 'yearbuilt_woe'] = bin1_woe
# data_wt_num.loc[data_wt_num.yearbuilt_bin == 2, 'yearbuilt_woe'] = bin2_woe
# data_wt_num.loc[data_wt_num.yearbuilt_bin == 3, 'yearbuilt_woe'] = bin3_woe
# data_wt_num.loc[data_wt_num.yearbuilt_bin == 4, 'yearbuilt_woe'] = bin4_woe
# data_wt_num.loc[data_wt_num.yearbuilt_bin == 5, 'yearbuilt_woe'] = bin5_woe
# data_wt_num.loc[data_wt_num.yearbuilt_bin == 6, 'yearbuilt_woe'] = bin6_woe
# data_wt_num.loc[data_wt_num.yearbuilt_bin == 7, 'yearbuilt_woe'] = bin7_woe
# data_wt_num.loc[data_wt_num.yearbuilt_bin == 8, 'yearbuilt_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.numstories_bin == 9, 'numstories_woe'] = bin9_woe

data_wt_num.head()


# In[75]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['staxvaldolcnt_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[76]:


bin0_woe = math.log(0.650092)
bin1_woe = math.log(0.894912)
bin2_woe = math.log(0.983281)
bin3_woe = math.log(1.029588)
bin4_woe = math.log(1.139155)
bin5_woe = math.log(1.155342)
bin6_woe = math.log(1.217046)
bin7_woe = math.log(1.219505)
bin8_woe = math.log(1.171286)
bin9_woe = math.log(0.860865)

data_wt_num.loc[data_wt_num.staxvaldolcnt_bin == 0, 'staxvaldolcnt_woe'] = bin0_woe
data_wt_num.loc[data_wt_num.staxvaldolcnt_bin == 1, 'staxvaldolcnt_woe'] = bin1_woe
data_wt_num.loc[data_wt_num.staxvaldolcnt_bin == 2, 'staxvaldolcnt_woe'] = bin2_woe
data_wt_num.loc[data_wt_num.staxvaldolcnt_bin == 3, 'staxvaldolcnt_woe'] = bin3_woe
data_wt_num.loc[data_wt_num.staxvaldolcnt_bin == 4, 'staxvaldolcnt_woe'] = bin4_woe
data_wt_num.loc[data_wt_num.staxvaldolcnt_bin == 5, 'staxvaldolcnt_woe'] = bin5_woe
data_wt_num.loc[data_wt_num.staxvaldolcnt_bin == 6, 'staxvaldolcnt_woe'] = bin6_woe
data_wt_num.loc[data_wt_num.staxvaldolcnt_bin == 7, 'staxvaldolcnt_woe'] = bin7_woe
data_wt_num.loc[data_wt_num.staxvaldolcnt_bin == 8, 'staxvaldolcnt_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.staxvaldolcnt_bin == 9, 'staxvaldolcnt_woe'] = bin9_woe

data_wt_num.head()


# In[77]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['taxvaldolcnt_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[78]:


bin0_woe = math.log(0.724694)
bin1_woe = math.log(0.824358)
bin2_woe = math.log(0.953287)
bin3_woe = math.log(1.056304)
bin4_woe = math.log(1.155673)
bin5_woe = math.log(1.239522)
bin6_woe = math.log(1.263289)
bin7_woe = math.log(1.243263)
bin8_woe = math.log(1.126373)
bin9_woe = math.log(0.769528)

data_wt_num.loc[data_wt_num.taxvaldolcnt_bin == 0, 'taxvaldolcnt_woe'] = bin0_woe
data_wt_num.loc[data_wt_num.taxvaldolcnt_bin == 1, 'taxvaldolcnt_woe'] = bin1_woe
data_wt_num.loc[data_wt_num.taxvaldolcnt_bin == 2, 'taxvaldolcnt_woe'] = bin2_woe
data_wt_num.loc[data_wt_num.taxvaldolcnt_bin == 3, 'taxvaldolcnt_woe'] = bin3_woe
data_wt_num.loc[data_wt_num.taxvaldolcnt_bin == 4, 'taxvaldolcnt_woe'] = bin4_woe
data_wt_num.loc[data_wt_num.taxvaldolcnt_bin == 5, 'taxvaldolcnt_woe'] = bin5_woe
data_wt_num.loc[data_wt_num.taxvaldolcnt_bin == 6, 'taxvaldolcnt_woe'] = bin6_woe
data_wt_num.loc[data_wt_num.taxvaldolcnt_bin == 7, 'taxvaldolcnt_woe'] = bin7_woe
data_wt_num.loc[data_wt_num.taxvaldolcnt_bin == 8, 'taxvaldolcnt_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.taxvaldolcnt_bin == 9, 'taxvaldolcnt_woe'] = bin9_woe

data_wt_num.head()


# In[79]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['ltaxvaldolcnt_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[80]:


bin0_woe = math.log(0.796621)
bin1_woe = math.log(0.896835)
bin2_woe = math.log(0.948642)
bin3_woe = math.log(1.00797)
bin4_woe = math.log(1.074838)
bin5_woe = math.log(1.165298)
bin6_woe = math.log(1.197719)
bin7_woe = math.log(1.229264)
bin8_woe = math.log(1.136947)
bin9_woe = math.log(0.772703)

data_wt_num.loc[data_wt_num.ltaxvaldolcnt_bin == 0, 'ltaxvaldolcnt_woe'] = bin0_woe
data_wt_num.loc[data_wt_num.ltaxvaldolcnt_bin == 1, 'ltaxvaldolcnt_woe'] = bin1_woe
data_wt_num.loc[data_wt_num.ltaxvaldolcnt_bin == 2, 'ltaxvaldolcnt_woe'] = bin2_woe
data_wt_num.loc[data_wt_num.ltaxvaldolcnt_bin == 3, 'ltaxvaldolcnt_woe'] = bin3_woe
data_wt_num.loc[data_wt_num.ltaxvaldolcnt_bin == 4, 'ltaxvaldolcnt_woe'] = bin4_woe
data_wt_num.loc[data_wt_num.ltaxvaldolcnt_bin == 5, 'ltaxvaldolcnt_woe'] = bin5_woe
data_wt_num.loc[data_wt_num.ltaxvaldolcnt_bin == 6, 'ltaxvaldolcnt_woe'] = bin6_woe
data_wt_num.loc[data_wt_num.ltaxvaldolcnt_bin == 7, 'ltaxvaldolcnt_woe'] = bin7_woe
data_wt_num.loc[data_wt_num.ltaxvaldolcnt_bin == 8, 'ltaxvaldolcnt_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.ltaxvaldolcnt_bin == 9, 'ltaxvaldolcnt_woe'] = bin9_woe

data_wt_num.head()


# In[81]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['taxamt_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[82]:


bin0_woe = math.log(0.793432)
bin1_woe = math.log(0.852044)
bin2_woe = math.log(0.964681)
bin3_woe = math.log(1.066847)
bin4_woe = math.log(1.177952)
bin5_woe = math.log(1.201814)
bin6_woe = math.log(1.199465)
bin7_woe = math.log(1.210905)
bin8_woe = math.log(1.058413)
bin9_woe = math.log(0.746259)

data_wt_num.loc[data_wt_num.taxamt_bin == 0, 'taxamt_woe'] = bin0_woe
data_wt_num.loc[data_wt_num.taxamt_bin == 1, 'taxamt_woe'] = bin1_woe
data_wt_num.loc[data_wt_num.taxamt_bin == 2, 'taxamt_woe'] = bin2_woe
data_wt_num.loc[data_wt_num.taxamt_bin == 3, 'taxamt_woe'] = bin3_woe
data_wt_num.loc[data_wt_num.taxamt_bin == 4, 'taxamt_woe'] = bin4_woe
data_wt_num.loc[data_wt_num.taxamt_bin == 5, 'taxamt_woe'] = bin5_woe
data_wt_num.loc[data_wt_num.taxamt_bin == 6, 'taxamt_woe'] = bin6_woe
data_wt_num.loc[data_wt_num.taxamt_bin == 7, 'taxamt_woe'] = bin7_woe
data_wt_num.loc[data_wt_num.taxamt_bin == 8, 'taxamt_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.taxamt_bin == 9, 'taxamt_woe'] = bin9_woe

data_wt_num.head()


# In[83]:


table = data_wt_num.pivot_table(values='weight', index=['pf'], columns=['censustract_bin'], aggfunc='sum', margins=True)
table2 = table.div(table.iloc[:,-1], axis=0)

df = table2.reset_index()
df = df.drop(2)
df = df.drop('All', axis=1)
df = df.drop('pf', axis=1)

df.divide(df.iloc[1])


# In[84]:


bin0_woe = math.log(0.949991)
bin1_woe = math.log(0.648737)
bin2_woe = math.log(1.011617)
bin3_woe = math.log(0.934396)
bin4_woe = math.log(0.935857)
bin5_woe = math.log(0.878347)
bin6_woe = math.log(1.504115)
bin7_woe = math.log(1.19881)
bin8_woe = math.log(1.156131)
bin9_woe = math.log(1.171952)

data_wt_num.loc[data_wt_num.censustract_bin == 0, 'censustract_woe'] = bin0_woe
data_wt_num.loc[data_wt_num.censustract_bin == 1, 'censustract_woe'] = bin1_woe
data_wt_num.loc[data_wt_num.censustract_bin == 2, 'censustract_woe'] = bin2_woe
data_wt_num.loc[data_wt_num.censustract_bin == 3, 'censustract_woe'] = bin3_woe
data_wt_num.loc[data_wt_num.censustract_bin == 4, 'censustract_woe'] = bin4_woe
data_wt_num.loc[data_wt_num.censustract_bin == 5, 'censustract_woe'] = bin5_woe
data_wt_num.loc[data_wt_num.censustract_bin == 6, 'censustract_woe'] = bin6_woe
data_wt_num.loc[data_wt_num.censustract_bin == 7, 'censustract_woe'] = bin7_woe
data_wt_num.loc[data_wt_num.censustract_bin == 8, 'censustract_woe'] = bin8_woe
data_wt_num.loc[data_wt_num.censustract_bin == 9, 'censustract_woe'] = bin9_woe

data_wt_num.head()


# In[85]:


data_wt_num.shape


# In[86]:


data_wt_num.to_csv("final_wts_2016.csv", index=False, encoding='utf8')

