
# coding: utf-8

# **Dealing with Missing Values**
# 
# One of the first steps in building a good predictive model is to carefully handle missing values at the start. 
# There's quite a lot of missing data in this dataset.
# 
# After investigating the data in some detail there also appears to be some fields which represent similar if not the same information which I think we can probably be remove as they are redundant. 
# 
# There are also potentially some inconsistent fields and potentially incorrect data.
# 
# The approaches use to deal with missing values also have to be made to the test data consistently.
# 
# Ideally after each step taken to deal with missing values you would probably want to carry out some cross-validation to see if it has helped improve your model.

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


rawdata= pd.read_csv("C:/Users/myang/Desktop/Zillow-Data-Analysis-master/Data/train_2016_v2.csv") 
rawdata.shape


# In[3]:


prop_df = pd.read_csv("C:/Users/myang/Desktop/Zillow-Data-Analysis-master/Data/properties_2016.csv")
prop_df.shape


# In[4]:


for c, dtype in zip(prop_df.columns, prop_df.dtypes):
    if dtype == np.float64:
        prop_df[c] = prop_df[c].astype(np.float32)

df_train = rawdata.merge(prop_df, how='left', on='parcelid')
#del prop_df, rawdataspecificrows
df_train.head(2)
#df_train.shape


# #### Pools & Hot tubs
# 
# There are actually multiple features related to pools:
# 
# - **"poolcnt"** - Number of pools on a lot. "NaN" means "0 pools", so we can update that to reflect "0" instead of "NaN".
# 
# 
# - **"hashottuborspa"** - Does the home have a hottub or a spa? "NaN" means "0 hottubs or spas", so we can update that to reflect "0" instead of "NaN".
# 
# 
# - **"poolsizesum"** - Total square footage of pools on property. Similarly, "NaN" means "0 sqare feet of pools", so we can also adjust that to read "0". For homes that do have pools, but are missing this information, we will just fill the "NaN" with the median value of other homes with pools.
# 
# 
# - **"pooltypeid2"** & **"pooltypeid7"** & **"pooltypeid10"** - Type of pool or hottub present on property. These categories will only contain non-null information if "poolcnt" or "hashottuborspa" contain non-null information. For the pool-related categories, we can fill the "NaN" value with a "0". And because "pooltypeid10" tells us the exact same information as "hashottuborspa", we can probably drop that category from our model.

# In[5]:


# "0 pools"
df_train.poolcnt.fillna(0,inplace = True)
df_train.head(2)


# In[6]:


# "0 hot tubs or spas"
df_train.hashottuborspa.fillna(0, inplace = True)
df_train.head(2)


# In[7]:


# Convert "True" to 1
df_train.hashottuborspa.replace(to_replace = True, value = 1,inplace = True)
df_train.head(2)


# In[8]:


print(df_train['hashottuborspa'].value_counts())


# In[9]:


# Set properties that have a pool but no info on poolsize equal to the median poolsize value.
df_train.loc[df_train.poolcnt==1, 'poolsizesum'] = df_train.loc[df_train.poolcnt==1, 'poolsizesum'].fillna(df_train[df_train.poolcnt==1].poolsizesum.median())

# "0 pools" = "0 sq ft of pools"
df_train.loc[df_train.poolcnt==0, 'poolsizesum']=0

# "0 pools with a spa/hot tub"
df_train.pooltypeid2.fillna(0,inplace = True)

# "0 pools without a hot tub"
df_train.pooltypeid7.fillna(0,inplace = True)

# Drop redundant feature
df_train.drop('pooltypeid10', axis=1, inplace=True)

df_train.head(2)


# #### Fireplace Data
# 
# There are two features related to fireplaces:
# 
# - **"fireplaceflag"** - Does the home have a fireplace? The answers are either "True" or "NaN". We will change the "True" values to "1" and the "NaN" values to "0".
# 
# - **"fireplacecnt"** - How many fireplaces in the home? We can replace "NaN" values with "0".
# 
# Looking deeper, it seems odd that over 10% of the homes have 1 or more fireplaces according to the "fireplacecnt" feature, but less than 1% of homes actually have "fireplaceflag" set to "True". There are obviously some errors with this data collection. To fix this, we will do the following:
# 
# - If "fireplaceflag" is "True" and "fireplacecnt" is "NaN", we will set "fireplacecnt" equal to the median value of "1".
# 
# - If "fireplacecnt" is 1 or larger "fireplaceflag" is "NaN", we will set "fireplaceflag" to "True".
# 
# We will change "True" in "fireplaceflag" to "1", so we can more easily analyze the information.

# In[10]:


print(df_train['fireplaceflag'].value_counts())


# In[11]:


# If "fireplaceflag" is "True" and "fireplacecnt" is "NaN", we will set "fireplacecnt" equal to the median value of "1".
df_train.loc[(df_train['fireplaceflag'] == True) & (df_train['fireplacecnt'].isnull()), ['fireplacecnt']] = 1


# In[12]:


# If 'fireplacecnt' is "NaN", replace with "0"
df_train.fireplacecnt.fillna(0,inplace = True)


# In[13]:


# If "fireplacecnt" is 1 or larger "fireplaceflag" is "NaN", we will set "fireplaceflag" to "True".
df_train.loc[(df_train['fireplacecnt'] >= 1.0) & (df_train['fireplaceflag'].isnull()), ['fireplaceflag']] = True
df_train.fireplaceflag.fillna(0,inplace = True)


# In[14]:


print(df_train['fireplaceflag'].value_counts())


# In[15]:


# Convert "True" to 1
df_train.fireplaceflag.replace(to_replace = True, value = 1,inplace = True)


# In[16]:


df_train.head(2)


# #### Garage Data
# 
# There are two features related to garages:
# 
# - **"garagecarcnt"** - How many garages does the house have? Easy fix here - we can replace "NaN" with "0" if a house doesn't have a garage.
# 
# 
# - **"garagetotalsqft"** - What is the square footage of the garage? Again, if a home doesn't have a garage, we can replace "NaN" with "0".
# 
# Unlike the Fireplace category where we have several Type II errors (false negative), we do not have any scenarios where a home has a "garagecarcnt" of "NaN", but a "garagetotalsqft" of some value.

# In[17]:


df_train.garagecarcnt.fillna(0,inplace = True)
df_train.garagetotalsqft.fillna(0,inplace = True)


# In[18]:


df_train.head(2)


# #### Tax Data Delinquency
# 
# There are two features related to tax delinquency:
# 
# - **"taxdelinquencyflag"** - Property taxes for this parcel are past due as of 2015.
# 
# 
# - **"taxdelinquencyyear"** - Year for which the unpaid property taxes were due.

# In[19]:


print(df_train['taxdelinquencyflag'].value_counts())


# In[20]:


# Replace "NaN" with "0"
df_train.taxdelinquencyflag.fillna(0,inplace = True)

# Change "Y" to "1"
df_train.taxdelinquencyflag.replace(to_replace = 'Y', value = 1,inplace = True)


# In[21]:


# Drop "taxdelinquencyyear" because it probably isn't a necessary variable
df_train.drop('taxdelinquencyyear', axis=1, inplace=True)


# In[22]:


df_train.head(2)


# #### The Rest
# 
# - **"storytypeid"** - Numerical ID that describes all types of homes. Mostly missing, so we should drop this category. Crazy idea would be to try and integrate street view of each home, and use image recognition to classify each type of story ID.

# In[23]:


# Drop "storytypeid"
df_train.drop('storytypeid', axis=1, inplace=True)


# - **"basementsqft"** - Square footage of basement. Mostly missing, suggesting no basement, so we will replace "NaN" with "0".

# In[24]:


# Replace "NaN" with 0, signifying no basement.
df_train.basementsqft.fillna(0,inplace = True)


# - **"yardbuildingsqft26"** - Storage shed square footage. We can set "NaN" values to "0". Might be useful to change this to a categorical category of just "1"s and "0"s (has a shed vs doesn't have a storage shed), but some of the sheds are enormous and others are tiny, so we will keep the actual square footage.

# In[25]:


# Replace 'yardbuildingsqft26' "NaN"s with "0".
df_train.yardbuildingsqft26.fillna(0,inplace = True)


# - **"architecturalstyletypeid"** - What is the architectural style of the house? Examples: ranch, bungalow, Cape Cod, etc. Because this is only present in a small fraction of the homes, I'm going to drop this category. (Idea: One can also assume that most homes in the same neighborhood have the same style. Could also try image recognition.)

# In[26]:


# Drop "architecturalstyletypeid"
df_train.drop('architecturalstyletypeid', axis=1, inplace=True)


# - **"typeconstructiontypeid"** - What material is the house made out of? Missing in a bunch, so probably drop category. Would be very difficult image recognition problem.
# 
# - **"finishedsquarefeet13"** - Perimeter of living area. This seems more like describing the shape of the house and is closely related to the square footage. I recommend dropping the category.

# In[27]:


# Drop "typeconstructiontypeid" and "finishedsquarefeet13"
df_train.drop('typeconstructiontypeid', axis=1, inplace=True)
df_train.drop('finishedsquarefeet13', axis=1, inplace=True)


# - **"buildingclasstypeid"** - Describes the internal structure of the home. Not a lot of information gained and present in less than 1% of properties. I will drop.

# In[28]:


# Drop "buildingclasstypeid"
df_train.drop('buildingclasstypeid', axis=1, inplace=True)


# In[29]:


print(df_train.shape)
df_train.notnull().mean().sort_values(ascending = False)


# - **"decktypeid"** - Type of deck (if any) on property. Looks like a value is either "66.0" or "NaN". I will keep this feature and change the "66.0" to "1" for "Yes" and "NaN" to "0" for "No".

# In[30]:


# Let's check the unique values for "decktypeid"
print(df_train['decktypeid'].value_counts())


# In[31]:


# Change "decktypeid" "Nan"s to "0"
df_train.decktypeid.fillna(0,inplace = True)
# Convert "decktypeid" "66.0" to "1"
df_train.decktypeid.replace(to_replace = 66.0, value = 1,inplace = True)


# In[32]:


print(df_train['decktypeid'].value_counts())
df_train.head(2)


# - **"finishedsquarefeet6"** - Base unfinished and finished area. Not sure what this means. Seems like it gives valuable information, but replacing "NaN"s with "0"s would be incorrect. Perhaps it is a subset of other categories. Probably drop, but TBD.
# 
# - **"finishedsquarefeet15"** - Total area. Should be equal to sum of all other finishedsquarefeet categories.
# 
# - **"finishedfloor1squarefeet"** - Sq footage of first floor. Could cross check this with number of stories.
# 
# - **"finishedsquarefeet50"** - Identical to above category? Drop one of them. Duplicate.
# 
# - **"finishedsquarefeet12"** - Finished living area.
# 
# - **"calculatedfinishedsquarefeet"** - Total finished living area of home.

# In[33]:


squarefeet = df_train[df_train['finishedsquarefeet15'].notnull() & 
                      df_train['finishedsquarefeet50'].notnull() & 
                      df_train['lotsizesquarefeet'].notnull()]
squarefeet[['calculatedfinishedsquarefeet','finishedsquarefeet6','finishedsquarefeet12','finishedsquarefeet15','finishedsquarefeet50','numberofstories','lotsizesquarefeet','landtaxvaluedollarcnt','structuretaxvaluedollarcnt','taxvaluedollarcnt','taxamount']]


# **"finishedsquarefeet6"** is rarely present, and even when it is present, it is equal to **"calculatedfinishedsquarefeet"**. Because of this, we will drop it. Same scenario with **"finishedsquarefeet12"**, so we will drop that as well. **"finishedsquarefeet50"** is identical to **"finishedfloor1squarefeet"**, so we will also drop **"finishedfloor1squarefeet"**.

# In[34]:


# Drop "finishedsquarefeet6"
df_train.drop('finishedsquarefeet6', axis=1, inplace=True)

# Drop "finishedsquarefeet12"
df_train.drop('finishedsquarefeet12', axis=1, inplace=True)

# Drop "finishedfloor1squarefeet"
df_train.drop('finishedfloor1squarefeet', axis=1, inplace=True)

df_train.head(2)


# In[35]:


squarefeet2 = df_train[df_train['finishedsquarefeet15'].notnull() & 
                            df_train['finishedsquarefeet50'].notnull() & 
                            df_train['lotsizesquarefeet'].notnull()]

squarefeet2[['calculatedfinishedsquarefeet','finishedsquarefeet15','finishedsquarefeet50','numberofstories','lotsizesquarefeet']]


# In[36]:


df_train.notnull().mean().sort_values(ascending = False)


# - **"calculatedfinishedsquarefeet"** - Present in 98%. Total finished living area of home. Let's fill the rest with the median values.
# 
# - **"finishedsquarefeet15"** - Present in 6.4%. Most cases, it is equal to **"calculatedfinishedsquarefeet"**, so we will fill in the "NaN" values with the value of "calculatedfinishedsquarefeet". Total area. Should be equal to sum of all other finishedsquarefeet categories.
# 
# - **"finishedsquarefeet50"** - If **"numberofstories"** is equal to "1", then we can replace the "NaN"s with the **"calculatedfinishedsquarefeet"** value. Fill in the rest with the average values.

# In[37]:


# Replace "NaN" "calculatedfinishedsquarefeet" values with mean.
df_train['calculatedfinishedsquarefeet'].fillna((df_train['calculatedfinishedsquarefeet'].mean()), inplace=True)


# In[38]:


# Replace "NaN" "finishedsquarefeet15" values with calculatedfinishedsquarefeet.
df_train.loc[df_train['finishedsquarefeet15'].isnull(),'finishedsquarefeet15'] = df_train['calculatedfinishedsquarefeet']


# In[39]:


df_train.numberofstories.fillna(1,inplace = True)


# In[40]:


print(df_train['numberofstories'].value_counts())


# In[41]:


# If "numberofstories" is equal to "1", then we can replace the "NaN"s with the "calculatedfinishedsquarefeet" value. Fill in the rest with the average values.
df_train.loc[df_train['numberofstories'] == 1.0,'finishedsquarefeet50'] = df_train['calculatedfinishedsquarefeet']
df_train['finishedsquarefeet50'].fillna((df_train['finishedsquarefeet50'].mean()), inplace=True)


# In[42]:


print(df_train.shape)
df_train.notnull().mean().sort_values(ascending = False)


# - **"yardbuildingsqft17"** - Patio in yard. Do same as storage shed category.

# In[43]:


# Replace 'yardbuildingsqft17' "NaN"s with "0".
df_train.yardbuildingsqft17.fillna(0,inplace = True)


# Now let's dig into the bathroom features.
# 
# - **"threequarterbathnbr"** - Number of 3/4 baths = shower, sink, toilet.
# 
# - **"fullbathcnt"** - Number of full bathrooms - tub, sink, toilet
# 
# - **"calculatedbathnbr"** - Total number of bathrooms including partials.
# 
# It seems like **"calculatedbathnbr"** should encompass the other two, so I will probably drop **"threequarterbathnbr"** and **"fullbathcnt"**, but let's take a look at some data first...

# In[44]:


bathrooms = df_train[df_train['fullbathcnt'].notnull() & 
                     df_train['threequarterbathnbr'].notnull() & 
                     df_train['calculatedbathnbr'].notnull()]
bathrooms[['fullbathcnt','threequarterbathnbr','calculatedbathnbr']]


# It looks like **"threequarterbathnbr"** is only a half-bath. Because **"calculatedbathnbr"** incorporates the other two, we will drop them. Then we will fill in the missing values for **"calculatedbathnbr"** with the most common answer.

# In[45]:


# Drop "threequarterbathnbr"
df_train.drop('threequarterbathnbr', axis=1, inplace=True)

# Drop "fullbathcnt"
df_train.drop('fullbathcnt', axis=1, inplace=True)

# Fill in "NaN" "calculatedbathnbr" with most common
bathroommode = df_train['calculatedbathnbr'].value_counts().idxmax()
df_train['calculatedbathnbr'] = df_train['calculatedbathnbr'].fillna(bathroommode)


# In[46]:


print(df_train.shape)
df_train.notnull().mean().sort_values(ascending = False)


# - **"airconditioningtypeid"** - If "NaN", change to "5" for "None".

# In[47]:


df_train.airconditioningtypeid.fillna(5,inplace = True)


# - **"regionidneighborhood"** - Neighborhood. Could fill in blanks. Would need a key that maps lat & longitude regions with specific neighborhoods. Because **"longitude"** and **"latitude"** essentially provide this information, we will drop **"regionidneighborhood"**.

# In[48]:


# Drop "regionidneighborhood"
df_train.drop('regionidneighborhood', axis=1, inplace=True)


# - **"heatingorsystemtypeid"** - Change "NaN" to "13" for "None" *REVIST THIS*

# In[49]:


df_train.heatingorsystemtypeid.fillna(13,inplace = True)


# - **"buildingqualitytypeid"** - Change "NaN" to most common value.

# In[50]:


print(df_train['buildingqualitytypeid'].value_counts())


# In[51]:


# Fill in "NaN" "buildingqualitytypeid" with most common
buildingqual = df_train['buildingqualitytypeid'].value_counts().idxmax()
df_train['buildingqualitytypeid'] = df_train['buildingqualitytypeid'].fillna(buildingqual)

df_train.head(2)


# - **"unitcnt"** - Number of units in a property. Change "NaN" to "1"

# In[52]:


df_train.unitcnt.fillna(1,inplace = True)


# - **"propertyzoningdesc"** - This seems like a very error-ridden column with so many unique values. It may provide some valuable info, so lets just fill the "NaN" with the most common value.

# print(df_train['propertyzoningdesc'].value_counts())

# In[53]:


# Fill in "NaN" "propertyzoningdesc" with most common
propertyzoningdesc = df_train['propertyzoningdesc'].value_counts().idxmax()
df_train['propertyzoningdesc'] = df_train['propertyzoningdesc'].fillna(propertyzoningdesc)


# - **"lotsizesquarefeet"** - Area of lot in square feet. Fill "NaN" with average value between 25 to 75 tax value Quantile *REVIST THIS*

# In[54]:


#lotsize_2575 = df_train.taxvaluedollarcnt.quartile(.25)

df_train['lotsizesquarefeet'].fillna((df_train['lotsizesquarefeet'].mean()), inplace=True)


# - **"censustractandblock"** & **"rawcensustractandblock"** - Census tract and block ID combined. Look like duplicate values. I think we should drop these because they are related to location which is covered by "longitude" and "latitude". Let's view the values first.

# In[55]:


# Drop "censustractandblock"
df_train.drop('censustractandblock', axis=1, inplace=True)


# - **"landtaxvaluedollarcnt"** - Assessed value of land area of parcel.
# 
# - **"structuretaxvaluedollarcnt"** - Assessed value of built structure on land.
# 
# - **"taxvaluedollarcnt"** - Total tax assessed value of property. "structuretax..." + "landtax...".
# 
# - **"taxamount"** - Total property tax assessed for assessment year.

# Let's filter our data and view the relationships of these columns. This should allow us to strategically fill in the blanks.

# In[56]:


taxdata = df_train[df_train['landtaxvaluedollarcnt'].notnull() & 
                        df_train['structuretaxvaluedollarcnt'].notnull() & 
                        df_train['taxvaluedollarcnt'].notnull() & 
                        df_train['taxamount'].notnull()]
taxdata[['landtaxvaluedollarcnt','structuretaxvaluedollarcnt','taxvaluedollarcnt','taxamount']]


# - **"landtaxvaluedollarcnt"** - We can fill in the "NaN"s with "0". It appears some properties do not have own any land. An example of this could be an apartment in large building where only the structurevalue would exist.
# 
# - **"structuretaxvaluedollarcnt"** - Same as **"landtaxvaluedollarcnt"**, but opposite. An example of a "NaN" in this category would be an empty lot.
# 
# - **"taxvaluedollarcnt"** - We can just fill in the "NaN" values with the average.
# 
# - **"taxamount"** - We should calculate a new category called *'taxpercentage'* where we divide the taxamount by the 'taxvaluedollarcnt', then we can fill in the "NaN" values with the average tax percentage.

# In[57]:


df_train.landtaxvaluedollarcnt.fillna(0,inplace = True)

df_train.structuretaxvaluedollarcnt.fillna(0,inplace = True)

df_train['taxvaluedollarcnt'].fillna((df_train['taxvaluedollarcnt'].mean()), inplace=True)


# In[58]:


df_train['taxpercentage'] = df_train['taxamount'] / df_train['taxvaluedollarcnt']
df_train.head(2)


# In[59]:


df_train['taxpercentage'].fillna((df_train['taxpercentage'].mean()), inplace=True)


# Now we will drop **"taxamount"** because we have replaced it with **"taxpercentage"**.

# In[60]:


# Drop "taxamount"
df_train.drop('taxamount', axis=1, inplace=True)


# - **"regionidcity"** - City property is located in. This is redundant information, so we will drop. *REVISIT THIS*

# In[61]:


# Drop "regionidcity"
df_train.drop('regionidcity', axis=1, inplace=True)


# - **"yearbuilt"** - Year home was built. We can just fill in the "NaN" values with the most common value.

# In[62]:


# Fill in "NaN" "yearbuilt" with most common
yearbuilt = df_train['yearbuilt'].value_counts().idxmax()
df_train['yearbuilt'] = df_train['yearbuilt'].fillna(yearbuilt)


# In[63]:


print(df_train.shape)
print(df_train.dtypes)
df_train.notnull().mean().sort_values(ascending = False)


# In[64]:


# Fill in "fips" "NaN"s
fips = df_train['fips'].value_counts().idxmax()
df_train['fips'] = df_train['fips'].fillna(fips)

# Fill in "propertylandusetypeid" "NaN"s
propertylandusetypeid = df_train['propertylandusetypeid'].value_counts().idxmax()
df_train['propertylandusetypeid'] = df_train['propertylandusetypeid'].fillna(propertylandusetypeid)

# Drop 'regionidcounty' - REVIST LATER
# df_train.drop('regionidcounty', axis=1, inplace=True)

# Fill in "latitude" "NaN"s
latitude = df_train['latitude'].value_counts().idxmax()
df_train['latitude'] = df_train['latitude'].fillna(latitude)

# Fill in "longitude" "NaN"s
longitude = df_train['longitude'].value_counts().idxmax()
df_train['longitude'] = df_train['longitude'].fillna(longitude)

# Fill in "rawcensustractandblock" "NaN"s
rawcensustractandblock = df_train['rawcensustractandblock'].value_counts().idxmax()
df_train['rawcensustractandblock'] = df_train['rawcensustractandblock'].fillna(rawcensustractandblock)

# Fill in "assessmentyear" "NaN"s
assessmentyear = df_train['assessmentyear'].value_counts().idxmax()
df_train['assessmentyear'] = df_train['assessmentyear'].fillna(assessmentyear)

# Fill in "bedroomcnt" "NaN"s
bedroomcnt = df_train['bedroomcnt'].value_counts().idxmax()
df_train['bedroomcnt'] = df_train['bedroomcnt'].fillna(bedroomcnt)

# Fill in "bathroomcnt" "NaN"s
bathroomcnt = df_train['bathroomcnt'].value_counts().idxmax()
df_train['bathroomcnt'] = df_train['bathroomcnt'].fillna(bathroomcnt)

# Fill in "roomcnt" "NaN"s
roomcnt = df_train['roomcnt'].value_counts().idxmax()
df_train['roomcnt'] = df_train['roomcnt'].fillna(roomcnt)

# Fill in "propertycountylandusecode" "NaN"s
propertycountylandusecode = df_train['propertycountylandusecode'].value_counts().idxmax()
df_train['propertycountylandusecode'] = df_train['propertycountylandusecode'].fillna(propertycountylandusecode)

# Fill in "regionidzip " "NaN"s
regionidzip = df_train['regionidzip'].value_counts().idxmax()
df_train['regionidzip'] = df_train['regionidzip'].fillna(regionidzip)


# In[65]:


df_train.to_csv("Test_Prop_2016.csv", index=False, encoding='utf8')

