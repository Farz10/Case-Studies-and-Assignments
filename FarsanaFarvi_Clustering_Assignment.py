#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries and reading data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("wine_clust.csv")


# ### Preprocessing

# In[3]:


df.head()


# In[4]:


df.describe()


# In[6]:


df.shape


# In[7]:


df.info()


# ### Finding null values and hamdling them if present

# In[8]:


df.isnull().sum()


# In[ ]:


#no null values are present in this dataset


# ### Correlation heatmap

# In[9]:


correlation = df.corr()
plt.subplots(figsize = (9,9))
sns.heatmap(correlation.round(2),annot = True)
plt.show()


# ### Standardisation

# In[10]:


from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(df)
X_scaled_array = scaler.transform(df)
wine_norm = pd.DataFrame(X_scaled_array, columns = df.columns)
#this standardised wine_norm dataset is used for clustering to so that attributes has common scale  
wine_norm.head()


# ### KMeans Clustering

# In[11]:


from sklearn.cluster import KMeans
# Calculating WCSS (within-cluster sums of squares) 
wcss=[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)
    kmeans.fit(wine_norm)
    wcss.append(kmeans.inertia_)


# ### Elbow plot(kmeans)

# In[12]:


plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')


# #### From the figure above, we can see that when the number of cluster is 3,it's hard to predict now whether adding another cluster can improve much better the inertia or not.

# In[13]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state =42)
y_kmeans = kmeans.fit_predict(wine_norm)
y_kmeans


# In[14]:


#finding with clusters = 4 to check score inorder to ensure inertia can be improved or not
kmeans1 = KMeans(n_clusters = 4, init = 'k-means++', random_state =42)
y_kmeans1 = kmeans1.fit_predict(wine_norm)
y_kmeans1


# ### Silhouette Score(kmeans)

# In[15]:


from sklearn.metrics import silhouette_score


# In[16]:


#for 3 clusters
sil_kmeans_3 =silhouette_score(wine_norm,y_kmeans)
sil_kmeans_3


# In[17]:


#for 4 clusters
sil_kmeans_4 =silhouette_score(wine_norm,y_kmeans1)
sil_kmeans_4


# ##### So its clear Kmeans CLustering with 3 clusters is the optimum option for this dataset since it gives higher silhouette score (0.2848589191898987)

# ### KMeans with PCA

# In[18]:


df1 = pd.read_csv("wine_clust.csv")
#importing standardscaler
from sklearn.preprocessing import StandardScaler
#satndardisation
scaler1 = StandardScaler()
scaled_data1 = scaler.fit_transform(df1)
scaled_data1 = pd.DataFrame(scaled_data1,columns = df1.columns)
scaled_data1


# In[20]:


### Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 13)
pca.fit(scaled_data1)


# In[21]:


x_pca = pca.transform(scaled_data1)
x_pca.shape


# In[23]:


### KMeans on PCA transformed data
#kmeans on pca tranformed data
kmeans_pca = KMeans(n_clusters=3)
kmeans_pca.fit(x_pca)
kmeans_labels = kmeans.predict(x_pca)


# In[24]:


# Visualize clusters
import matplotlib.pyplot as plt
plt.scatter(x_pca[:,0], x_pca[:,1], c=kmeans_labels)
plt.title('K-means clustering with PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


# In[25]:


###checking silhouette score for kmeans clustering with pca transformed data
#3 clusters
sil_kmeans_pca =silhouette_score(x_pca,kmeans_labels)
sil_kmeans_pca


# ### Agglomerative Clustering

# In[32]:


import scipy.cluster.hierarchy as sch
#draw dendrogram
dendrogram = sch.dendrogram(sch.linkage(wine_norm, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()


# ###### can see 3 clusters is correct choice since longest dendrite cuts at 3 points.

# In[33]:


from sklearn.cluster import AgglomerativeClustering
ahc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_ahc = ahc.fit_predict(df)
y_ahc


# ### Silhouette Score (AHC)

# In[35]:


silhouette_scores = []
for i in range(2, 13):
    ac = AgglomerativeClustering(n_clusters=i)
    y_ac = ac.fit_predict(wine_norm)
    silhouette_scores.append(silhouette_score(wine_norm, y_ac))
silhouette_scores


# ###### Hence 3 clusters is the optimum cluster option for agglomerative clustering due to its silhouette score (0.2774439826952265).

# ### AC with PCA

# In[37]:


ahc_pca = AgglomerativeClustering(n_clusters=3)
ahc_pca.fit(x_pca)
labels_ac_pca = ahc_pca.fit_predict(x_pca)


# In[38]:


# Visualize clusters
import matplotlib.pyplot as plt
plt.scatter(x_pca[:,0], x_pca[:,1], c=labels_ac_pca)
plt.title('Agglomerative clustering with PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


# ### Silhouette score (AHC with PCA)

# In[39]:


sil_ac_pca =silhouette_score(x_pca,labels_ac_pca)
sil_ac_pca


# ##### Score is same for both cases, hence can choose 3 clusters as optimum option in this dataset for Agglomerative clustering (0.2774439826952265).

# ### DB Scan Clustering

# In[41]:


#For the eps parameter, we'll set it to 0.5, and for min_samples, we'll set it to 5. 
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_clusters = dbscan.fit_predict(wine_norm)


# In[42]:


#Let's check how many clusters were created:
df['cluster']= dbscan.labels_
print(df['cluster'].value_counts())


# In[44]:


# Performing PCA
X = df.iloc[:, 1:].values
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


# In[45]:


plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_clusters)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


# ###### gets only one single cluster from this dataset in dbscan clustering

# #### Hence so far kmeans cluster with 3 clusters is optimum choice for this dataset.

# In[ ]:




