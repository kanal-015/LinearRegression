#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import math
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd


# In[16]:


data=pd.read_csv('Downloads\concrete.csv')
print(data)


# In[6]:



c=data.columns
print(c)


# In[7]:


X=data[['cementcomp', 'slag', 'flyash', 'water', 'superplastisizer',
       'coraseaggr', 'finraggr', 'age']]
print(X)
Y=data[['CCS']]
print(Y)


# In[8]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,train_size=0.7)
X_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)


# In[9]:


model=sm.OLS(Y_train,X_train).fit()
predict_data=model.predict(X_test)
print(predict_data)


# In[10]:


fig,ax=plt.subplots(2)
ax[0].plot(X_test,Y_test,'bx',label='True')
ax[1].plot(X_test,predict_data,'ro',label='OLS prediction')


# In[15]:


model=LinearRegression()#these all steps are used to calculate and print accuracy of our model
model.fit(X_train,Y_train)
acc=model.score(X_test,Y_test)
print(acc)


# In[ ]:




