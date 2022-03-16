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


# # question 2

# In[15]:


import pandas as pd
import scipy 
from scipy import stats
import statsmodels.api as sm


# In[17]:


data=pd.read_csv("LabTAT.csv")
data


# In[18]:


stats.f_oneway(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2],data.iloc[:,3])


# In[ ]:


# comparing pvalue 
0.02<0.05,Accept Alternative hypothesis 
we say that, Average of atleast 1 laboratory are different


# In[ ]:


#question 3


# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import scipy 
import numpy as np
from scipy import stats


# In[21]:


df=pd.read_csv("BuyerRatio.csv")
df


# In[22]:


df_table=df.iloc[:,1:5]
df_table


# In[23]:


plt.plot(df_table.East)


# In[24]:


plt.plot(df_table.West)


# In[25]:


plt.plot(df_table.North)


# In[26]:


plt.plot(df_table.South)


# In[27]:


Chisquares_results=scipy.stats.chi2_contingency(df_table)
Chisquares_results


# In[ ]:


#comparing p value
0.660>0.05, accept null hypothesis 
we conclude that, All proportions are equal


# In[ ]:


#question 4


# In[28]:


import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency


# In[29]:


data=pd.read_csv("Costomer+OrderForm.csv")
data


# In[30]:


data.Phillippines.value_counts()


# In[31]:


data.Malta.value_counts()


# In[32]:


data.India.value_counts()


# In[33]:


data.Indonesia.value_counts()


# In[34]:


order=np.array([[271,267,269,280],[29,33,31,20]])
order


# In[35]:


customer=chi2_contingency(order)
customer


# In[ ]:


inference: pvalue(0.27710)>0.05 so we accept the nul hypothesis

