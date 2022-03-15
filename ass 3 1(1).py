#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import scipy 
import numpy as np
from scipy import stats 


# In[3]:


df=pd.read_csv("Cutlets.csv")
df


# In[4]:


plt.hist(df["Unit A"])


# In[5]:


plt.hist(df['Unit B'])


# In[12]:


unita=pd.Series(df.iloc[:,0])
unita


# In[13]:


unitb=pd.Series(df.iloc[:,1])
unitb


# In[14]:


stats.ttest_ind(unita,unitb) 


# In[ ]:


# comparing p value with 0.05
0.472>0.05, so,accept null hypothesis 
we conclude that, there is no difference in diameters of cutlets between two units

