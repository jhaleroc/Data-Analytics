#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt

dataset_train = pd.read_csv('el4233-2018-2019-02-klasifikasi-train.csv',sep=',',header=0)
dataset_test = pd.read_csv('el4233-2018-2019-02-klasifikasi-test.csv',sep=',',header=0)
dataset_saz = pd.read_csv('el4233-2018-2019-02-klasifikasi-submit-all-zeros.csv',sep=',',header=0)

dataset_train.head()


# In[2]:


dataset_test.head()


# In[3]:


dataset_saz.head()


# In[ ]:



    


# In[4]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset_train[['X0', 'X1']], dataset_train[['Y']], test_size=0.2, random_state=58)


# In[5]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import PolynomialFeatures

data_train_mse = 100 
derajat_polinom = 5
#Logistic Regression
for derajat_polinom in range(1,10):
    poly_features = PolynomialFeatures(degree=derajat_polinom, include_bias=True)
    data_x_train_poly = poly_features.fit_transform(x_train)
    data_x_test_poly = poly_features.fit_transform(x_test)
    
    logistic_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs',  n_jobs=5)
    logistic_reg.fit(data_x_train_poly, y_train)
    test_y_predict=logistic_reg.predict(data_x_test_poly)
    data_train_mse_new = mse(y_test, test_y_predict)
    #mencari derajat polinom optimal dan mse terkecil
    if data_train_mse_new<data_train_mse:
        data_train_mse = data_train_mse_new
        dp_optimal = derajat_polinom
        
        print(dp_optimal)
        print(data_train_mse)
            


# In[6]:


poly_features = PolynomialFeatures(degree=5, include_bias=True)

data_x_train_poly = poly_features.fit_transform(x_train)
data_x_test_poly = poly_features.fit_transform(x_test)

logistic_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs',  n_jobs=5)
logistic_reg.fit(data_x_train_poly, y_train)

data_y_test_pred=logistic_reg.predict(data_x_test_poly)
data_train_mse_new = mse(y_test, data_y_test_pred)
x_test_data = dataset_test[['X0','X1']]
test_data_x_test_poly = poly_features.fit_transform(x_test_data)
test_data_y_test_pred = logistic_reg.predict(test_data_x_test_poly)

print(data_train_mse_new)
print(test_data_x_test_poly)
print(test_data_y_test_pred)


# In[52]:


output_saz = dataset_saz
output_saz['prediksi'] = test_test_y_predict
np.savetxt("out_logistic_reg.csv", output_saz, delimiter=",", fmt="%s", header="No, X0, X1, prediksi")
df_out=pd.read_csv('out_logistic_reg.csv', sep=',',header=0)
df_out.head()


# In[7]:


from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

x_train, x_test, y_train, y_test = train_test_split(dataset_train[['X0','X1']], dataset_train[['Y']], test_size=0.2, random_state=58)

data_train_mse = 100
#Decision Tree
for i in range(1,5):
    for j in range(100):
        ran_forest = RandomForestClassifier(n_estimators=i+2, n_jobs=6, random_state=j)
        ran_forest.fit(x_train,y_train)
        test_y_predict = ran_forest.predict(x_test)
        data_train_mse_new = mse(y_test, test_y_predict)
        if data_train_mse_new<data_train_mse:
            data_train_mse = data_train_mse_new
            optimal = i+2
            
            print(j)  
            print(optimal)
            print(data_train_mse)
            


# In[54]:


#Random Forest

ran_forest = RandomForestClassifier(n_estimators=3, n_jobs=6, random_state=52)
ran_forest.fit(x_train,y_train)
test_y_predict = ran_forest.predict(x_test)

data_train_mse_new = mse(y_test, test_y_predict)
ran_forest = RandomForestClassifier(n_estimators=3, n_jobs=6, random_state=52)
ran_forest.fit(x_train,y_train)


# In[55]:


data_test_y_predict=dt.predict(dataset_test[['X0','X1']])
output_saz = dataset_test
output_saz['prediksi'] = data_test_y_predict
np.savetxt("out_random_forest.csv", output_saz, delimiter=",", fmt="%s", header="No, X0, X1, prediksi")
df_out=pd.read_csv('out_random_forest.csv', sep=',',header=0)
df_out.head()


# In[ ]:




