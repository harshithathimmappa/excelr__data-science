#!/usr/bin/env python
# coding: utf-8

# In[2]:


#question 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


x=pd.Series([24.23,25.53,25.41,24.14,29.62,28.25,25.81,24.39,40.26,32.95,91.36,25.99,39.42,26.71,35.00])


# In[6]:


name=['Allied Signal','Bankers Trust','General Mills','ITT Industries','J.P.Morgan & Co.','Lehman Brothers',
      'Marriott','MCI','Merrill Lynch','Microsoft','Morgan Stanley','Sun Microsystems','Travelers','US Airways',
      'Warner-Lambert']


# In[7]:


# Pie Plot
plt.figure(figsize=(6,8))
plt.pie(x,labels=name,autopct='%1.0f%%')
plt.show()


# In[8]:


# Box Plot to find outliars
sns.boxplot(x)


# In[9]:


# Mean
x.mean()


# In[10]:


# Vairance
x.var()


# In[11]:


# Standard Deviation
x.std()


# In[ ]:




