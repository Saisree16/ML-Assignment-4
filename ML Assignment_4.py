#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")


# In[3]:


df=pd.read_csv("C:\\Users\\19139\\Downloads\\Salary_Data.csv.xls")
df.head(5)


# In[4]:


X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, test_size=1/3,random_state = 0)


# In[7]:


X.shape


# In[8]:


X_Train.shape


# In[9]:


X_Test.shape


# In[5]:


regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)

Y_Pred = regressor.predict(X_Test)


# In[6]:


mean_squared_error(Y_Test,Y_Pred)


# In[10]:


plt.title('Training data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.scatter(X_Train, Y_Train)
plt.show()


# In[11]:


plt.title('Testing data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.scatter(X_Test, Y_Test)
plt.show()


# In[15]:


df2=pd.read_csv("C:\\Users\\19139\\Downloads\\K-Mean_Dataset.csv")
df2.head()


# In[16]:



X = df2.iloc[:,1:].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X)
X = imputer.transform(X)


# In[18]:


wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()


# In[20]:


from sklearn.cluster import KMeans
nclusters = 4 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(X)


# In[23]:


y_cluster_kmeans = km.predict(X)
from sklearn import metrics
score = metrics.silhouette_score(X, y_cluster_kmeans)
print('Silhouette score:',score)


# In[ ]:


scaler = preprocessing.StandardScaler()
scaler.fit(X)
X_scaled_array = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled_array, columns = X.columns)


# In[29]:


from sklearn.cluster import KMeans
nclusters = 4 
km = KMeans(n_clusters=nclusters)
km.fit(X_scaled_array)


# In[32]:


y_scaled_cluster_kmeans = km.predict(X_scaled_array)
from sklearn import metrics
score = metrics.silhouette_score(X_scaled_array, y_scaled_cluster_kmeans)
print('Silhouette score after applying scaling:',score)

