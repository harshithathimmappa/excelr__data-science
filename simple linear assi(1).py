#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#simple linear 


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


sal=pd.read_csv("Salary_Data.csv")
sal


# In[5]:


sal.info()


# In[6]:


sal.describe()


# In[7]:


import seaborn as sns
sns.distplot(sal['YearsExperience'])


# In[8]:


sns.distplot(sal['Salary'])


# In[9]:


sal.corr()


# In[10]:


import seaborn as sns
sns.regplot(sal.YearsExperience,sal.Salary,"bo")
plt.ylabel("SALARY")
plt.xlabel("years of experience")


# In[11]:


import statsmodels.formula.api as smf
model=smf.ols("Salary~YearsExperience",data=sal).fit()
model.params


# In[12]:


model.rsquared


# In[13]:


model.summary()


# In[14]:


model.predict(sal)


# In[15]:


Salary=(25792.200199)+(9449.962321)*1.1
Salary


# In[ ]:




