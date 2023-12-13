
#importing kmeans DBSCAN HAC matplotlib and numpy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd 
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

# read in our data
cc_data  = pd.read_csv('PythonZenva/ClusterAnalysis/clustering/CC_GENERAL.csv', index_col=False)
# print nulls
#print(cc_data.isnull().sum())

#print(cc_data)

#plt.scatter(cc_data["CREDIT_LIMIT"],cc_data["BALANCE"])
#plt.show()

# sample size - 1000
#sample1000 = cc_data.sample(1000)

#plt.scatter(sample1000["CREDIT_LIMIT"],sample1000["BALANCE"])
#plt.show()

# draw elbow method graph 
subsample = cc_data.sample(1000, random_state=17)

X = pd.concat([subsample["CREDIT_LIMIT"], subsample["BALANCE"]], axis=1)

wcss = []
for k in range (1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.scatter(range (1, 11), wcss)
plt.show()

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# plot
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# plot
plt.scatter(X["CREDIT_LIMIT"],X["BALANCE"], c=kmeans.labels_,cmap='viridis')

# prevent elongates axis
plt.axis('equal')
plt.colorbar()
plt.show()