#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.formula.api as smf
from  statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


data=pd.read_csv("50_Startups.csv")
data


# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


data=data.rename({'R&D Spend':'RDS','Administration':'ADMI','Marketing Spend':'MKS'},axis=1)
data


# In[6]:


data.describe()


# In[7]:


data.isna().sum()


# In[8]:


data.corr()


# In[9]:


sns.set_style(style="darkgrid")
sns.pairplot(data)


# In[10]:


model=smf.ols('Profit~MKS+ADMI+RDS',data=data).fit()


# In[11]:


model.params


# In[12]:


(model.rsquared, model.rsquared_adj)


# In[13]:


model.summary()


# In[14]:


ml_m=smf.ols('Profit~MKS',data = data).fit()  
print(ml_m.tvalues, '\n', ml_m.pvalues) 


# In[15]:


ml_m.summary()


# In[16]:


ml_ad=smf.ols('Profit~ADMI',data=data).fit()
print(ml_ad.tvalues,  '\n', ml_ad.pvalues)


# In[17]:


ml_ad.summary()


# In[18]:


ml_mad=smf.ols('Profit~MKS+ADMI',data=data).fit()
print(ml_mad.tvalues,  '\n', ml_mad.pvalues)


# In[19]:


ml_mad.summary()


# In[20]:


rsq_r=smf.ols("RDS~ADMI+MKS",data=data).fit().rsquared
vif_r=1/(1-rsq_r)

rsq_a=smf.ols("ADMI~RDS+MKS",data=data).fit().rsquared
vif_a=1/(1-rsq_a)

rsq_m=smf.ols("MKS~RDS+ADMI",data=data).fit().rsquared
vif_m=1/(1-rsq_m)


# In[21]:


d1={'Variables':['RDS','ADMS','MKTS'],'Vif':[vif_r,vif_a,vif_m]}
Vif_df=pd.DataFrame(d1)
Vif_df


# In[22]:


import statsmodels.api as sm
qqplot=sm.qqplot(model.resid,line='q')
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[23]:


list(np.where(model.resid<-30000)) 


# In[24]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()


# In[26]:


plt.scatter(get_standardized_values(model.fittedvalues),
            get_standardized_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# In[27]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "RDS", fig=fig)
plt.show()


# In[28]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "ADMI", fig=fig)
plt.show()


# In[29]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "MKS", fig=fig)
plt.show()


# In[30]:


model_influence = model.get_influence()
(c,_)=model.get_influence().cooks_distance
c


# In[31]:


fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(data)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[32]:


np.argmax(c) , np.max(c)


# In[33]:


influence_plot(model)
plt.show()


# In[34]:


k=data.shape[1]
n=data.shape[0]
leverage_cutoff = (3*(k+1))/n
leverage_cutoff


# In[35]:


data[data.index.isin([49])] 


# In[36]:


data.head()


# In[37]:


data1=data.drop(data.index[49],axis=0).reset_index()
data1.head()


# In[38]:


data1=data1.drop(['index'],axis=1)
data1.head()


# In[39]:


final_mod= smf.ols('Profit~MKS+ADMI+RDS',data = data1).fit()


# In[40]:


(final_mod.rsquared,final_mod.aic)


# In[41]:


data1


# In[42]:


final_mod.rsquared


# In[43]:


pred=final_mod.predict(data1)
pred


# In[ ]:


# question 2 toyoto


# In[44]:


import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot


# In[45]:


toyo=pd.read_csv("ToyotaCorolla.csv",encoding='latin1')
toyo


# In[46]:


toyo.shape


# In[47]:


toyo.info()


# In[48]:


toyo1=pd.concat([toyo.iloc[:,2:4],toyo.iloc[:,6:7],toyo.iloc[:,8:9],toyo.iloc[:,12:14],toyo.iloc[:,15:18]],axis=1)
toyo1


# In[49]:


toyo1.shape


# In[50]:


toyo2=toyo1.rename({'Age_08_04':'Age','cc':'CC','Quarterly_Tax':'QT'},axis=1)
toyo2


# In[51]:


toyo2[toyo2.duplicated()]


# In[52]:


toyo3=toyo2.drop_duplicates().reset_index(drop=True)
toyo3


# In[53]:


toyo3.describe()


# In[54]:


toyo3.corr()


# In[55]:


sns.set_style(style='dark')
sns.pairplot(toyo3)


# In[56]:


model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=toyo3).fit()


# In[57]:


model.params


# In[58]:


model.summary()


# In[59]:


mlr_c=smf.ols('Price~CC',data=toyo3).fit()
mlr_c.tvalues , mlr_c.pvalues


# In[60]:


mlr_c=smf.ols('Price~Doors',data=toyo3).fit()
mlr_c.tvalues , mlr_c.pvalues


# In[61]:


mlr_c=smf.ols('Price~Doors+CC',data=toyo3).fit()
mlr_c.tvalues , mlr_c.pvalues


# In[62]:


rsq_age=smf.ols('Age~KM+HP+CC+Doors+Gears+QT+Weight',data=toyo3).fit().rsquared
vif_age=1/(1-rsq_age)

rsq_KM=smf.ols('KM~Age+HP+CC+Doors+Gears+QT+Weight',data=toyo3).fit().rsquared
vif_KM=1/(1-rsq_KM)

rsq_HP=smf.ols('HP~Age+KM+CC+Doors+Gears+QT+Weight',data=toyo3).fit().rsquared
vif_HP=1/(1-rsq_HP)

rsq_CC=smf.ols('CC~Age+KM+HP+Doors+Gears+QT+Weight',data=toyo3).fit().rsquared
vif_CC=1/(1-rsq_CC)

rsq_DR=smf.ols('Doors~Age+KM+HP+CC+Gears+QT+Weight',data=toyo3).fit().rsquared
vif_DR=1/(1-rsq_DR)

rsq_GR=smf.ols('Gears~Age+KM+HP+CC+Doors+QT+Weight',data=toyo3).fit().rsquared
vif_GR=1/(1-rsq_GR)

rsq_QT=smf.ols('QT~Age+KM+HP+CC+Doors+Gears+Weight',data=toyo3).fit().rsquared
vif_QT=1/(1-rsq_QT)

rsq_WT=smf.ols('Weight~Age+KM+HP+CC+Doors+Gears+QT',data=toyo3).fit().rsquared
vif_WT=1/(1-rsq_WT)


# In[63]:


d1={'Variables':['Age','KM','HP','CC','Doors','Gears','QT','Weight'],
    'VIF':[vif_age,vif_KM,vif_HP,vif_CC,vif_DR,vif_GR,vif_QT,vif_WT]}
Vif_df=pd.DataFrame(d1)
Vif_df


# In[64]:


import statsmodels.api as sm
sm.qqplot(model.resid,line='q') 
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[65]:


list(np.where(model.resid>4000)) 


# In[66]:


list(np.where(model.resid<-4500))


# In[67]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()


# In[69]:


plt.scatter(get_standardized_values(model.fittedvalues),
            get_standardized_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# In[70]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Age", fig=fig)
plt.show()


# In[71]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Doors", fig=fig)
plt.show()


# In[72]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "KM", fig=fig)
plt.show()


# In[73]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "HP", fig=fig)
plt.show()


# In[74]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "CC", fig=fig)
plt.show()


# In[75]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Gears", fig=fig)
plt.show()


# In[76]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "QT", fig=fig)
plt.show()


# In[77]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Weight", fig=fig)
plt.show()


# In[78]:


model_influence = model.get_influence()
(c,_)=model.get_influence().cooks_distance
c


# In[79]:


fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(toyo3)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[80]:


(np.argmax(c),np.max(c))


# In[81]:


influence_plot(model)
plt.show()


# In[82]:


k=toyo3.shape[1]
n=toyo3.shape[0]
leverage_cutoff = (3*(k+1))/n
leverage_cutoff


# In[83]:


toyo3[toyo3.index.isin([80])] 


# In[84]:


toyo_new=toyo3.copy()
toyo_new


# In[85]:


toyo3.shape


# In[86]:


toyo4=toyo_new.drop(toyo_new.index[[80]],axis=0).reset_index(drop=True)


# In[87]:


toyo4


# In[88]:


toyo4.shape


# In[89]:


final_model1=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=toyo4).fit()


# In[90]:


(final_model1.rsquared,final_model1.aic)


# In[91]:


final_model1.summary()


# In[92]:


model_influence_V =final_model1.get_influence()
(c_V, _) =model_influence_V.cooks_distance


# In[93]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(toyo4)),np.round(c_V,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance');


# In[94]:


(np.argmax(c_V),np.max(c_V))


# In[95]:


toyo_new1=toyo4.copy()
toyo_new1


# In[96]:


toyo4.shape


# In[97]:


toyo5=toyo4.drop(toyo4.index[[219,220]],axis=0).reset_index(drop=True)


# In[98]:


toyo5


# In[99]:


toyo5.shape


# In[100]:


final_model2=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=toyo5).fit()


# In[101]:


(final_model2.rsquared,final_model2.aic)


# In[102]:


final_model2.summary()


# In[103]:


model_influence_V1 = final_model2.get_influence()
(c_V1, _) = model_influence_V1.cooks_distance


# In[104]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(toyo5)),np.round(c_V1,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance');


# In[105]:


(np.argmax(c_V1),np.max(c_V1))


# In[106]:


toyo_new2=toyo5.copy()
toyo_new2


# In[107]:


toyo6=toyo5.drop(toyo5.index[956],axis=0).reset_index(drop=True)


# In[108]:


toyo6


# In[109]:


final_modelfix=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=toyo6).fit()


# In[110]:


(final_modelfix.rsquared,final_modelfix.aic)


# In[111]:


final_modelfix.summary()


# In[112]:


pred_y = final_modelfix.predict(toyo6)
pred_y


# In[ ]:




