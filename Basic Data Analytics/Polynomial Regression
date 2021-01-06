#!/usr/bin/env python
# coding: utf-8

# In[3]:


#simple ML
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Common imports
import numpy as np

# to make this notebook's output stable across runs
np.random.seed(42)


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd

#load the data
dataset_train = pd.read_csv('el4233-2018-2019-02-01-training.csv')
dataset_test = pd.read_csv('el4233-2018-2019-02-01-test.csv')


# In[5]:


dataset_train.head()


# In[6]:


dataset_test.head()


# In[7]:


x_test = pd.DataFrame(dataset_test.x).values
x_test


# In[8]:


from sklearn.model_selection import train_test_split
y = dataset_train.y
x = dataset_train.drop('y', axis=1)
x_train, x_validation, y_train, y_validation = train_test_split(x,y, test_size=0.2)
print("x_train =",  x_train.shape)
print("x_validation =",  x_validation.shape)
print("y_train =",  y_train.shape)
print("y_validation =",  y_validation.shape)


# In[9]:


from matplotlib import pyplot as plt

plt.plot(x_train, y_train, "go") 
plt.plot(x_validation, y_validation, "bo") 
print("Training Set = Green  Validation Set = Blue")
plt.xlabel("$x$", fontsize=16)
plt.ylabel("$y$", rotation=0, fontsize=16)
plt.axis([0, 2, -3, 3])
plt.show()


# In[10]:


derajat_polinom=25
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=derajat_polinom, include_bias=False)
x_train_poly_feat = poly_features.fit_transform(x_train)
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()
linear_reg.fit(x_train_poly_feat, y_train)
linear_reg.intercept_, linear_reg.coef_


# In[11]:


y_train_poly_feat=linear_reg.predict(x_train_poly_feat)
y_train_poly_feat.shape
x_test_poly_feat=poly_features.fit_transform(x_test)
x_test_poly_feat


# In[12]:


prediction_data = linear_reg.predict(x_test_poly_feat)
prediction_data


# In[14]:


plt.plot(x_train, y_train, "go") 
plt.plot(x_validation, y_validation, "bo") 
plt.plot(x_test, prediction_data, "rx") 

print("Training Set = Green Validation Set = Blue Prediction = Red")
plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$y$", rotation=0, fontsize=16)
plt.axis([0, 2, -3, 3])
plt.show()


# In[20]:


result=dataset_test
result['prediction_data']= prediction_data
print(result)
result.to_csv('prediksi_y_terhadap_x2.csv', sep=',',index=False,header=False)


# In[ ]:




