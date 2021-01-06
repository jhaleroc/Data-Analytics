#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division, print_function, unicode_literals
# Common imports
import numpy as np
import os

# agar outputnya selalu sama setiap dilakukan run kernel
np.random.seed(42)

# agar tidak mengeluarkan warning
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "tugas_akhir3"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#load the data
dt = pd.read_csv('data lengkap kai untuk ta.csv')
#dataset_test = pd.read_csv('el4233-2018-2019-02-01-test.csv')
#data lengkap kai untuk ta.csv
#Keterlambatan seluruh kereta - Dari KAI - 12 Kereta.csv


# In[3]:


dt.head()


# In[4]:


z=0
while (z < len(dt)):
    if(dt.Lokasi_1[z] == '-'):
        dt.Lokasi_1[z] = dt.Lokasi_2[z]
    z+=1


# In[5]:


dt.iloc[98]


# In[6]:



asd = np.savetxt("cekdata.csv", dt, delimiter=",", fmt="%s", header="Penyebab, Akibat, Lokasi_1,Lokasi_2,Andil,Bulan Ke,Minggu ke,No Kereta,Prediksi")
asd


# In[7]:


dt.describe()


# In[8]:


df = dt.drop(['Tanggal', 'Hari Ke','# No', 'Andil3', 'Andil2', 'Andil Absolut'], axis=1)
df.head()


# In[9]:


df_c = df.dropna()
df_c.describe()


# In[10]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df_c, test_size=0.2, random_state=42)

x_train = train_set.drop('Andil', axis = 1)
y_train = train_set.Andil

x_test = test_set.drop('Andil', axis = 1)
y_test = test_set.Andil


# In[11]:


x_train.head()


# In[12]:


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
column_trans = make_column_transformer(
    (OneHotEncoder(sparse = False,handle_unknown='ignore'), 
     ['Lokasi_1','Lokasi_2', 'No Kereta', 'Minggu Ke', 'Bulan Ke','Penyebab', 'Akibat', 'Nama Hari']),
    remainder='passthrough')


# In[13]:


from sklearn.tree import DecisionTreeRegressor
tree_clf = DecisionTreeRegressor(random_state=42)

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=30, max_features=8, random_state=42)

from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]
forest_reg2 = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg2, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)

from sklearn.svm import SVR
svm_reg = SVR(kernel="linear")

from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)

from sklearn.ensemble import VotingRegressor
voting_clf = VotingRegressor(
    estimators=[('fr', forest_reg), ('tr', tree_clf), ('svc', svm_reg), ('gb', gbrt)])


# In[14]:


from sklearn.pipeline import make_pipeline
pipe = make_pipeline(column_trans, forest_reg)
pipe2 = make_pipeline(column_trans, svm_reg)
pipe3 = make_pipeline(column_trans, tree_clf)
pipe4 = make_pipeline(column_trans, grid_search)
pipe5 = make_pipeline(column_trans, voting_clf)
pipe6 = make_pipeline(column_trans, gbrt)

pipe.fit(x_train, y_train)
pipe2.fit(x_train, y_train)
pipe3.fit(x_train, y_train)
pipe4.fit(x_train, y_train)
pipe5.fit(x_train, y_train)
pipe6.fit(x_train, y_train)


# In[ ]:





# In[15]:


y_pred = pipe5.predict(x_test)
from sklearn.metrics import mean_squared_error
tree_mse = mean_squared_error(y_test, y_pred)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[16]:


from sklearn.metrics import mean_absolute_error
y_pred = pipe5.predict(x_test)
lin_mae = mean_absolute_error(y_test, y_pred)
lin_mae


# In[17]:


from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(pipe2,x_train,y_train,cv=5)
cv_scores.mean()


# In[18]:


grid_search.best_params_


# In[19]:


grid_search.best_estimator_


# In[20]:


from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(pipe3,x_train,y_train,cv=5)
cv_scores.mean()


# In[21]:


ohe = OneHotEncoder(sparse=False, categories='auto')


# In[22]:


ohe.fit_transform(df_c[['No Kereta']])


# In[23]:


corr_matrix = df_c.corr()
corr_matrix["Andil"].sort_values(ascending=False)


# In[24]:


from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(pipe,x_train,y_train,cv=5)
cv_scores.mean()


# In[25]:


from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(pipe5,x_train,y_train,cv=5)
cv_scores.mean()


# In[26]:


from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(pipe6,x_train,y_train,cv=5)
cv_scores.mean()


# In[ ]:





# In[27]:


from scipy import stats
confidence = 0.95
squared_errors = (y_pred - y_test) ** 2
mean = squared_errors.mean()
m = len(squared_errors)

np.sqrt(stats.t.interval(confidence, m - 1,
                         loc=np.mean(squared_errors),
                         scale=stats.sem(squared_errors)))


# In[28]:


confidence2 = 0.99
np.sqrt(stats.t.interval(confidence2, m - 1,
                         loc=np.mean(squared_errors),
                         scale=stats.sem(squared_errors)))


# In[29]:


df.info


# In[30]:


dt["Penyebab"].value_counts()


# In[31]:


dt.describe()


# In[32]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
dt.hist(bins=50, figsize=(20,15))
plt.show()


# In[33]:


abc = test_set
#df = dt.drop(['Tanggal','Andil','Penyebab', 'Hari Ke', 'Nama Hari', '# No', 'Akibat'], axis=1)
abc['Prediksi'] = y_pred
np.savetxt("outpred.csv", abc, delimiter=",", fmt="%s", header="Penyebab, Akibat, Lokasi_1,Lokasi_2,Andil,Bulan Ke,Minggu ke,No Kereta,Prediksi")
abc


# In[34]:


df_out=pd.read_csv('outpred.csv', sep=',',header=0)
df_out


# In[35]:


list(df_out.columns) 


# In[36]:


#melakukan prediksi terhadap data input baru dengan model yang sudah dibuat 
dt_new = pd.read_csv('data_baru_untuk_diprediksi.csv')
dt_new_drop = dt_new.drop(['Tanggal','Andil2', 'Andil3', 'Hari Ke', '# No', 'Andil Absolut'], axis=1)
dt_new_cleaned = dt_new_drop.dropna()

x_new = dt_new_cleaned.drop('Andil', axis = 1)
y_new = dt_new_cleaned.Andil

y_new_pred = pipe5.predict(x_new)
dt_new_cleaned['Prediksi'] = y_new_pred
dt_new_cleaned


# In[ ]:





# In[37]:


y_new_pred


# In[38]:


y_new


# In[39]:


df_c["Akibat"].value_counts()


# In[40]:


df_c["Penyebab"].value_counts()


# In[41]:


dt_Pe = df_c[df_c.Penyebab=='PEMASANGAN TASPAT (PRASARANA)']
dt_M = df_c[df_c.Penyebab=='TUNGGU PERSILANGAN']
#dt_L = df_c[df_c.Lokasi_2=='Bandung']
dt_P = pd.concat([dt_Pe, dt_M])


# In[42]:


dt_P


# In[43]:


train_set2, test_set2 = train_test_split(dt_P, test_size=0.2, random_state=42)

x_train2 = train_set2.drop('Andil', axis = 1)
y_train2 = train_set2.Andil

x_test2 = test_set2.drop('Andil', axis = 1)
y_test2 = test_set2.Andil


# In[44]:


pipe10 = make_pipeline(column_trans, voting_clf)
pipe10.fit(x_train2, y_train2)


# In[45]:


from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(pipe10,x_train2,y_train2,cv=5)
cv_scores.mean()


# In[46]:


y_pred2 = pipe10.predict(x_test2)
from sklearn.metrics import mean_squared_error
tree_mse2 = mean_squared_error(y_test2, y_pred2)
tree_rmse2 = np.sqrt(tree_mse2)
tree_rmse2


# In[47]:


tree_mse2


# In[48]:


np.savetxt("y_pred2.csv", y_pred2, delimiter=",", fmt="%s")
y_pred2


# In[49]:


np.savetxt("y_test2.csv", y_test2, delimiter=",", fmt="%s")
y_test2.head()


# In[50]:


y_test2.describe()


# In[51]:


y_test2.head()


# In[52]:


from sklearn.metrics import mean_absolute_error

lin_mae2 = mean_absolute_error(y_test2, y_pred2)
lin_mae2


# In[53]:


from scipy import stats
confidence = 0.68
squared_errors = (y_pred2 - y_test2) ** 2
mean = squared_errors.mean()
m = len(squared_errors)

np.sqrt(stats.t.interval(confidence, m - 1,
                         loc=np.mean(squared_errors),
                         scale=stats.sem(squared_errors)))


# In[54]:


confidence = 0.95
squared_errors = (y_pred2 - y_test2) ** 2
mean = squared_errors.mean()
m = len(squared_errors)

np.sqrt(stats.t.interval(confidence, m - 1,
                         loc=np.mean(squared_errors),
                         scale=stats.sem(squared_errors)))


# In[ ]:





# In[ ]:




