#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division, print_function, unicode_literals
# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "tugas_akhir_tanggal"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(42)
#load the data
dt = pd.read_csv('kereta api - rangkuman Data kereta 396a.csv')
dt.head()


# In[3]:


z = 0
array_no_kereta = []
while (z < len(dt)):
    no_kereta = 395
    array_no_kereta.append(no_kereta) 
    z+=1
    
dt['No Kereta'] = array_no_kereta 


# In[4]:


z=0
while (z < len(dt)):
    if(dt.Lokasi_1[z] == '-'):
        dt.Lokasi_1[z] = dt.Lokasi_2[z]
    z+=1


# In[5]:


dk = dt.drop(['Daop', 'KM', 'Keterangan'], axis=1)
df = dk.dropna()


# In[6]:


df.Tanggal.dtype


# In[7]:


df.Tanggal = pd.to_datetime(df['Tanggal'])


# In[8]:


df


# In[9]:


type(df.Tanggal[0])


# In[10]:


#https://github.com/dc-aichara/DS-ML-Public/blob/master/Medium_Files/Date_Time.ipynb


print('Month :',df.Tanggal[0].month) # Get month
print('Day of Month :',df.Tanggal[0].day) # Get day of month
print('Day of Week :',df.Tanggal[0].weekday()) # Get weekday 0-6 --> Monday to Sunday
print(df.Tanggal[0].week)
#print('Hour :',df.Tanggal[0].hour)  # Get hour 
#print('Minutes :',df.Tanggal[0].minute) # Get minutes
#print('Seconds :',df.Tanggal[0].second) # Get Seconds


# In[11]:


array_month = [df.Tanggal[0].month]
array_hari = [df.Tanggal[0].day]
array_nama_hari = [df.Tanggal[0].weekday()]
array_minggu_ke = [df.Tanggal[0].week]
z=1
while (z < len(df)):
    bulan = df.Tanggal[z].month
    array_month.append(bulan)
    hari = df.Tanggal[z].day
    array_hari.append(hari)
    nama_hari = df.Tanggal[z].weekday()
    array_nama_hari.append(nama_hari)
    minggu_ke = df.Tanggal[z].week
    array_minggu_ke.append(minggu_ke)
    z+=1
    


# In[12]:


df['Bulan Ke'] = array_month 
df['Hari_Ke'] = array_hari
df['Nama Hari'] = array_nama_hari 
df['Minggu Ke'] = array_minggu_ke


# In[ ]:





# In[13]:


#date = datetime(2019,10,4) # create a datetime object
#for x in range(6):
#ketika dia pindah tanggal, mulai lagi dari 0
#andil_tiga = andil dua yang dijadikan x untuk prediksi andil stasiun berikutnya
j = 0
i = 0
k = 0
array_andil_bulan = []
array_andil_hari = []
array_andil_per_hari = []
array_andil_nama_hari = []
array_andil_minggu_ke = []
array_andil_dua = [df.Andil[0]]
array_andil_tiga = [df.Andil[0]]
date_sekarang = df.Tanggal[0] 
cek_date = df.Tanggal[0]
while (i < len(df)): #tanggal yang terkahir belum bisa masuk
    while (date_sekarang == cek_date) :
        j += 1
        i+=1
        if(j<len(df)):
            date_sekarang = df.Tanggal[j] 
            cek_date = df.Tanggal[j-1]
            if (date_sekarang == cek_date):
                andil_tiga = array_andil_dua[j - 1]
                array_andil_tiga.append(andil_tiga)
                andil_dua = df.Andil[j] + array_andil_tiga[j] #andil 3
                array_andil_dua.append(andil_dua)
        else:
            break
    if(i<len(df)):#untuk masuk ke tanggal berikutnya
        array_andil_per_hari.append(andil_dua)#menghitung keterlambatan per hari
        bulan_sekarang = df.Tanggal[j-1].month
        tanggal_sekarang = df.Tanggal[j-1].day
        nama_hari_sekarang = df.Tanggal[j-1].weekday()
        minggu_ke_sekarang = df.Tanggal[j-1].week
        array_andil_bulan.append(bulan_sekarang)
        array_andil_hari.append(tanggal_sekarang)
        array_andil_nama_hari.append(nama_hari_sekarang)
        array_andil_minggu_ke.append(minggu_ke_sekarang)
        ##################################################
        date_sekarang = df.Tanggal[j] #untuk pindah tanggal
        cek_date = df.Tanggal[j]
        nilai_tiga = df.Andil[j]
        array_andil_tiga.append(nilai_tiga)
        nilai_dua = df.Andil[j]  #andil 3
        array_andil_dua.append(nilai_dua)
        k=k+1


# In[14]:


array_andil_bulan


# In[15]:


dt_andil = pd.DataFrame({
   'Hari Ke': array_andil_hari,
   'Bulan Ke': array_andil_bulan,
    'Nama_Hari': array_andil_nama_hari,
    'Minggu_Ke':array_andil_minggu_ke,
    'Andil': array_andil_per_hari
})


# In[16]:


dt_andil.head()


# In[17]:


array_andil_nama_hari


# In[18]:


array_andil_tiga


# In[19]:


i


# In[20]:


len(df)


# In[21]:


df['Andil2'] = array_andil_dua


# In[22]:


df['Andil3'] = array_andil_tiga


# In[23]:


df.head()


# In[24]:


df.out = np.savetxt("out_395.csv", df, delimiter=",", fmt="%s", header="No,Tanggal,Penyebab,Akibat,Lokasi 1,Lokasi 2,Andil,No Kereta,Bulan Ke,Hari Ke,Nama Hari,Minggu Ke,Andil2,Andil3")


# In[25]:


df_out=pd.read_csv('out_395.csv', sep=',',header=0)
df_out.head()


# In[26]:


df_corr = df.corr()
df_corr["Andil2"].sort_values(ascending = False)


# In[27]:


df_corr["Andil"].sort_values(ascending = False)


# In[28]:


dt["Penyebab"].value_counts()


# In[29]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
dt_andil.hist(bins=50, figsize=(20,15))
plt.show()


# In[30]:


from sklearn.preprocessing import OrdinalEncoder
housing_cat = df_out[['Lokasi 2']]
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]


# In[31]:


ordinal_encoder.categories_


# In[32]:


df_out['Encoded_Lokasi_2'] = housing_cat_encoded


# In[33]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
df_out.hist(bins=50, figsize=(20,15))
plt.show()


# In[34]:


df_out

