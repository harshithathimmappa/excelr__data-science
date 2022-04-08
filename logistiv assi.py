#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#logistic


# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report 


# In[2]:


df=pd.read_csv("bank-full.csv",sep=';')
df


# In[3]:


df.shape


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


df.info


# In[7]:


df.columns
df['outcome']=df.y.map({'no':0,'yes':1})
df


# In[8]:


sns.boxplot(x='outcome',y='age',data=df)


# In[9]:


sns.countplot(x="outcome",data=df,palette="hls") 


# In[10]:


dummies=pd.get_dummies(df,columns=['job','marital','education','contact','poutcome','month'])
dummies


# In[11]:


pd.set_option("display.max.columns", None)
dummies


# In[12]:


dummies.default=dummies.default.map(dict(yes=1,no=0))
dummies


# In[13]:


dummies.housing=dummies.housing.map(dict(yes=1,no=0))
dummies


# In[14]:


dummies.loan=dummies.loan.map(dict(yes=1,no=0))
dummies


# In[15]:


dummies.info()


# In[16]:


x=pd.concat([dummies.iloc[:,0:10],dummies.iloc[:,12:]],axis=1)
y=dummies.iloc[:,11]


# In[17]:


classifier=LogisticRegression()
classifier.fit(x,y)


# In[18]:


classifier.coef_


# In[19]:


classifier.predict_proba (x) 


# In[20]:


y_pred = classifier.predict(x)
dummies["y_pred"] = y_pred
dummies  


# In[21]:


y_prob = pd.DataFrame(classifier.predict_proba(x.iloc[:,:]))
new_df = pd.concat([dummies,y_prob],axis=1)
new_df  


# In[22]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y,y_pred)
print (confusion_matrix) 


# In[23]:


pd.crosstab(y_pred,y)  


# In[24]:


y_pred=classifier.predict(x)
y_pred


# In[25]:


accuracy = sum(y==y_pred)/dummies.shape[0]
accuracy 


# In[26]:


from sklearn.metrics import classification_report 
print (classification_report (y, y_pred)) 


# In[27]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
Logit_roc_score=roc_auc_score(y,classifier.predict(x))
Logit_roc_score 


# In[28]:


fpr, tpr, thresholds = roc_curve(y,classifier.predict_proba(x)[:,1]) 
plt.plot(fpr, tpr, label='Logistic Regression (area=%0.2f)'% Logit_roc_score)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[29]:


y_prob1 = pd.DataFrame(classifier.predict_proba(x)[:,1]) 


# In[30]:


y_prob1 


# In[31]:


import statsmodels.api as sm   
logit = sm.Logit(y, x)


# In[32]:


logit.fit().summary()  


# In[33]:


fpr 


# In[34]:


tpr


# In[ ]:




