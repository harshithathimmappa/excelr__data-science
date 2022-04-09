#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# question1 airlines


# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


airline=pd.read_excel("EastWestAirlines.xlsx",sheet_name='data')
airline.head()


# In[5]:


airline.info()


# In[6]:


airline2=airline.drop(['ID#'],axis=1)
airline2


# In[7]:


def norm_func(i):
         x=(i-i.min())/(i.max()-i.min())
         return (x)


# In[8]:


df_norm=norm_func(airline2.iloc[:,:])
df_norm


# In[9]:


from sklearn.preprocessing import MinMaxScaler
trans = MinMaxScaler()
airline3 = pd.DataFrame(trans.fit_transform(airline2.iloc[:,:]))
airline3 


# In[15]:


from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch  
z = linkage(df_norm, method="complete",metric="euclidean")
plt.figure(figsize=(10, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(z)  
plt.show()


# In[16]:


from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=5, linkage='complete',affinity = "euclidean").fit(df_norm) 

cluster_labels=pd.Series(h_complete.labels_)
cluster_labels
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline


# In[17]:


airline.iloc[:,1:].groupby(airline.clust).mean()


# In[18]:


#K MEANS
from sklearn.cluster import KMeans
east=pd.read_excel("EastWestAirlines.xlsx",sheet_name='data')
east.head()


# In[19]:


east2=east.drop(['ID#'],axis=1)
east2


# In[20]:


east2.info()


# In[21]:


def norm_func(i):
         x=(i-i.min())/(i.max()-i.min())
         return (x)


# In[22]:


east_norms=norm_func(east2.iloc[:,1:])
east_norms.head()


# In[23]:


fig = plt.figure(figsize=(10, 8))
WCSS = []
for i in range(1, 11):
    clf = KMeans(n_clusters=i)
    clf.fit(east_norms)
    WCSS.append(clf.inertia_) 
plt.plot(range(1, 11), WCSS)
plt.title('The Elbow Method')
plt.ylabel('WCSS')
plt.xlabel('Number of Clusters')
plt.show()


# In[24]:


clf = KMeans(n_clusters=4)
y_kmeans = clf.fit_predict(east_norms)


# In[25]:


y_kmeans


# In[26]:


md=pd.Series(y_kmeans) 
east['clust']=md  
east


# In[27]:


east.iloc[:,1:7].groupby(east.clust).mean()


# In[28]:


east.plot(x="Balance",y ="Qual_miles",c=clf.labels_,kind="scatter",s=50 ,cmap=plt.cm.coolwarm) 
plt.title('Clusters using KMeans')


# In[32]:


#db scan


# In[33]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# In[34]:


df=pd.read_excel("EastWestAirlines.xlsx",sheet_name='data')
df.head()


# In[35]:


df.info()


# In[36]:


df.describe()


# In[37]:


df1 = df.drop(['ID#'],axis=1) 
df1


# In[38]:


array=df1.values
array


# In[39]:


stscaler = StandardScaler().fit(array)
X = stscaler.transform(array) 
X 


# In[40]:


dbscan = DBSCAN(eps=0.45, min_samples=10)
dbscan.fit(X)


# In[41]:


dbscan.labels_ 


# In[42]:


cl=pd.DataFrame(dbscan.labels_,columns=['cluster']) 


# In[43]:


cl
pd.set_option("display.max_rows", None)


# In[44]:


cl  


# In[45]:


df1 = pd.concat([df,cl],axis=1) 
df1 


# In[46]:


import matplotlib.pyplot as plt
plt.style.use('classic')


# In[47]:


plt.figure(figsize=(10, 7))  
plt.scatter(df1['cluster'],df1['Balance'], c=dbscan.labels_) 


# In[48]:


dl = dbscan.labels_ 


# In[49]:


import sklearn
sklearn.metrics.silhouette_score(X, dl)


# In[50]:


from sklearn.cluster import KMeans
clf = KMeans(n_clusters=3)
y_kmeans = clf.fit_predict(X) 


# In[51]:


y_kmeans


# In[52]:


cl1=pd.DataFrame(y_kmeans,columns=['Kcluster']) 
cl1 


# In[53]:


sklearn.metrics.silhouette_score(X, y_kmeans)


# In[54]:


#questuin 2 crime


# In[55]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[56]:


crime = pd.read_csv("crime_data.csv")
crime.head()


# In[57]:


new=crime.rename(columns={'Unnamed: 0' : 'States'})
new


# In[58]:


new.describe()


# In[59]:


new.info()


# In[60]:


def norm_func(i):
         x=(i-i.min())/(i.max()-i.min())
         return (x)


# In[61]:


df_norm=norm_func(new.iloc[:,1:])
df_norm.head()


# In[62]:


from sklearn.preprocessing import MinMaxScaler
trans = MinMaxScaler()
data2 = pd.DataFrame(trans.fit_transform(new.iloc[:,1:]))
data2 


# In[63]:


from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch  
z = linkage(df_norm, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  
    leaf_font_size=8.,
    )
plt.show()


# In[64]:


from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=6, linkage='complete',affinity = "euclidean").fit(df_norm) 

cluster_labels=pd.Series(h_complete.labels_)
cluster_labels
new['clust']=cluster_labels 
new


# In[65]:


new.iloc[:,1:].groupby(new.clust).mean()


# In[66]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
new_subset = pd.DataFrame(scaler.fit_transform(new.iloc[:,1:]))
new_subset 


# In[67]:


from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch  
z = linkage(new_subset, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  
    leaf_font_size=8.,  
)
plt.show()


# In[68]:


from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=6, linkage='complete',affinity = "euclidean").fit(new_subset) 

cluster_labels=pd.Series(h_complete.labels_)
cluster_labels
new['clust']=cluster_labels 
new.head() 


# In[69]:


#K MEANS
from sklearn.cluster import KMeans
crime=pd.read_csv("crime_data.csv")
crime.head()


# In[70]:


def norm_func(i):
         x=(i-i.min())/(i.max()-i.min())
         return (x)


# In[71]:


df_norms=norm_func(crime.iloc[:,1:])
df_norms.head()


# In[72]:


fig = plt.figure(figsize=(10, 8))
WCSS = []
for i in range(1, 11):
    clf = KMeans(n_clusters=i)
    clf.fit(df_norms)
    WCSS.append(clf.inertia_) 
plt.plot(range(1, 11), WCSS)
plt.title('The Elbow Method')
plt.ylabel('WCSS')
plt.xlabel('Number of Clusters')
plt.show()


# In[73]:


clf = KMeans(n_clusters=4)
y_kmeans = clf.fit_predict(df_norms)


# In[74]:


y_kmeans


# In[75]:


md=pd.Series(y_kmeans) 
crime['clust']=md  
crime


# In[76]:


crime.iloc[:,1:7].groupby(crime.clust).mean()


# In[77]:


crime.plot(x="Murder",y ="Assault",c=clf.labels_,kind="scatter",s=50 ,cmap=plt.cm.coolwarm) 
plt.title('Clusters using KMeans')


# In[78]:


clf.inertia_


# In[79]:


WCSS


# In[80]:


#DB SCAN
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# In[81]:


crime1=pd.read_csv("crime_data.csv")
crime1.head()


# In[82]:


crime1.info()


# In[83]:


crime1.drop(['Unnamed: 0'],axis=1,inplace=True)
crime1


# In[84]:


array=crime1.values
array


# In[85]:


stscaler = StandardScaler().fit(array)
X = stscaler.transform(array) 
X 


# In[86]:


dbscan = DBSCAN(eps=1, min_samples=4)
dbscan.fit(X)


# In[87]:


dbscan.labels_ 


# In[88]:


cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])


# In[89]:


cl
pd.set_option("display.max_rows", None)


# In[90]:


cl


# In[91]:


crime2 = pd.concat([crime,cl],axis=1) 
crime2


# In[92]:


import matplotlib.pyplot as plt
plt.style.use('classic')


# In[93]:


crime2.plot(x="Murder",y ="Assault",c=dbscan.labels_ ,kind="scatter",s=50 ,cmap=plt.cm.copper_r) 
plt.title('Clusters using DBScan') 


# In[94]:


plt.figure(figsize=(10, 7))  
plt.scatter(crime2['cluster'],crime2['UrbanPop'], c=dbscan.labels_)


# In[95]:


crime2.groupby('cluster').agg(['mean']).reset_index()


# In[96]:


crime2 = dbscan.labels_ 


# In[97]:


import sklearn
sklearn.metrics.silhouette_score(X, crime2)


# In[ ]:




