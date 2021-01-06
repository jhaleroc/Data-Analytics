#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df_out=pd.read_csv('out_395.csv', sep=',',header=0)
df_no_minus = df_out


# In[2]:


df_no_minus.head()


# In[3]:


j=0
while (j < len(df_no_minus)):
    if(df_no_minus.Andil[j] < 0):
        df_no_minus.drop([j], axis=0, inplace=True)
    j+=1


# In[4]:


from sklearn.preprocessing import OrdinalEncoder
train_cat = df_out[['Lokasi 2']]
ordinal_encoder = OrdinalEncoder()
train_cat_encoded = ordinal_encoder.fit_transform(train_cat)
train_cat_encoded[:10]
df_out['Encoded_Lokasi_2'] = train_cat_encoded


# In[5]:


df_no_minus.describe()


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
df_no_minus.hist(bins=50, figsize=(20,15))
plt.show()


# In[7]:


df_no_minus["Encoded_Lokasi_2"].hist()


# In[8]:


df_no_minus["Encoded_Lokasi_2"].value_counts()


# In[9]:


ordinal_encoder.categories_


# In[ ]:




