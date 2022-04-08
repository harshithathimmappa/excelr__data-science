#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#simple linear question 1


# In[ ]:





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


# question 2


# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[18]:


data=pd.read_csv("delivery_time.csv")
data


# In[19]:


data.info()


# In[20]:


data.describe()


# In[21]:


data.isna().sum()


# In[22]:


plt.hist(data["Delivery Time"])


# In[23]:


plt.hist(data["Sorting Time"])


# In[24]:


data=data.rename({'Delivery Time':'delivery_time','Sorting Time':'sorting_time'},axis=1)
data


# In[25]:


data.corr()


# In[26]:


sns.regplot(x="sorting_time",y="delivery_time",data=data)


# In[27]:


import statsmodels.formula.api as smf
model=smf.ols("delivery_time~sorting_time",data=data).fit()
model.params


# In[28]:


model.summary()


# In[29]:


pred=model.predict(data)
pred


# In[30]:


model.resid


# In[31]:


model.resid_pearson


# In[32]:


#RMSE METHOD
pd.set_option("display.max_rows", 21) 
pred
rmse_lin = np.sqrt(np.mean((np.array(data['delivery_time'])-np.array(pred))**2))
rmse_lin 


# In[33]:


plt.scatter(x=data['sorting_time'],y=data['delivery_time'],color='red')
plt.plot(data['sorting_time'],pred,color='black')
plt.xlabel('SORTING TIME')
plt.ylabel('DELIVERY TIME')


# In[34]:


model2 = smf.ols('delivery_time~np.log(sorting_time)',data=data).fit()
model2.params


# In[35]:


model2.summary()


# In[36]:


pred1 = model2.predict(pd.DataFrame(data['sorting_time']))

pred1


# In[37]:


rmse_log = np.sqrt(np.mean((np.array(data['delivery_time'])-np.array(pred1))**2))
rmse_log 


# In[38]:


pred1.corr(data.delivery_time)


# In[39]:


plt.scatter(x=data['sorting_time'],y=data['delivery_time'],color='green')
plt.plot(data['sorting_time'],pred1,color='blue')
plt.xlabel('sorting time')
plt.ylabel('delivery time')


# In[40]:


model3 = smf.ols('np.log(delivery_time)~sorting_time',data=data).fit()


# In[41]:


model3.summary()


# In[42]:


pred_log = model3.predict(pd.DataFrame(data['sorting_time']))


# In[43]:


pred_log


# In[44]:


pred2=np.exp(pred_log)
pred2


# In[ ]:




