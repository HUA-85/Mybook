#!/usr/bin/env python
# coding: utf-8

# ## 機器學習工作流程
# 房價迴歸預測

# In[1]:


from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


ds = datasets.load_boston()
print(ds.DESCR)


# In[3]:


print(ds.feature_names) #特徵
#print(ds.target) #目標


# In[4]:


print(ds.target) #目標


# In[5]:


import pandas as pd
X = pd.DataFrame(ds.data,columns=ds.feature_names)
X.head()


# In[6]:


y = ds.target


# In[7]:


#a.isnull().sum()
X_train, X_test, y_train, y_test = train_test_split(X, y) #分割資料
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[8]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 

my_model = LinearRegression()
my_model.fit(X_train , y_train)


# ### RMSE

# In[9]:


import numpy as np
y_train_predict = my_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train,y_train_predict)))


# In[10]:


print("Regression of RMSE is {}".format(rmse) ,"for traning。")


# ###  R2 score

# In[11]:


R2 = r2_score(y_train,y_train_predict)
print("Regression of R2 score is {}".format(R2) ,"for traning。")


# In[ ]:




