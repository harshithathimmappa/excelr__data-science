#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 


# In[4]:


wine = pd.read_csv("wine.csv")
wine


# In[5]:


wine1 = wine.iloc[:,1:]
wine1


# In[6]:


wine1.shape


# In[7]:


wine1.describe()


# In[8]:


wine1.info()


# In[9]:


wine2= wine1.values
wine2


# In[10]:


wine_normal = scale(wine2)
wine_normal


# In[11]:


pca = PCA(n_components = 13)
pca_values = pca.fit_transform(wine_normal)
pca_values 


# In[12]:


pca.components_


# In[13]:


var = pca.explained_variance_ratio_
var


# In[14]:


var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1


# In[15]:


plt.plot(var1,color="red")


# In[16]:


finalDf = pd.concat([pd.DataFrame(pca_values[:,0:3],columns=['pc1','pc2','pc3']),wine['Type']], axis = 1)
finalDf


# In[17]:



plt.style.use('classic')


# In[18]:


import seaborn as sns
fig=plt.figure(figsize=(14,12))
sns.scatterplot(data=finalDf) 


# In[20]:


#hcluster
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch


# In[21]:


z = linkage(wine_normal, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(
    z,
    #leaf_rotation=0.,  
    #leaf_font_size=8.,  
)
plt.show()


# In[22]:


from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=3, linkage='complete',affinity = "euclidean").fit(wine_normal)
h_complete


# In[23]:


cluster_labels=pd.Series(h_complete.labels_)
cluster_labels


# In[24]:


wine['clusterid']=cluster_labels 
wine


# In[25]:


#kmean cluster
from sklearn.cluster import KMeans


# In[26]:


fig = plt.figure(figsize=(10, 8))
WCSS = []
for i in range(1,6 ):
    clf = KMeans(n_clusters=i)
    clf.fit(wine_normal)
    WCSS.append(clf.inertia_) 
plt.plot(range(1, 6), WCSS)
plt.title('The Elbow Method')
plt.ylabel('WCSS')
plt.xlabel('Number of Clusters')
plt.show()


# In[27]:


clf = KMeans(n_clusters=3)
y_kmeans = clf.fit_predict(wine_normal)


# In[28]:


y_kmeans
clf.labels_


# In[29]:


md=pd.Series(y_kmeans) 
wine1['clusterid2']=md 
wine1


# In[ ]:




