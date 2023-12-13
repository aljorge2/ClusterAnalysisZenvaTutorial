# Hierarchical Agglomerative Clustering (HAC) is another clustering algorithm at our disposal. 

# the notion of hierarchical clustering is that we build a tree / some kind of hierarchy of these clusters. A tree structure is a nice human
#    interpretable visualization.

# in agglomerative clustering, each point is initially its own cluster and we group them recursively. In divisive clustering, we go in the
    # opposite direction – all point are 1 cluster and we split them recursively. A similarity matrix is used t omerge clusters initially.

# each spilt in the tree structure is a cluster of data and the farther you go back the more section become a grouping until your are at the
#   root where all splits form one grouping.

# the only parameter used in HAC is the similarity matrix. In any other application of HAC, we might need to specify the number of clusters
    # to get concrete cluster groups. However it is not necessary for drawing the dendogram.

###########################
# Pseudo code HAC
###########################

# intially assign each point to each cluster

# find the 2 closest cluster using the similarity metric and merge them

# repeat second step until all clusters are merged into one cluster 

######################
###################### 

# HAC is really customizable and your own similiarity matrix can be used if needed

# for HAC we do need to specify the number of clusters. Based on the number points maybe assigned in the following ways. 

    # minimize cluster variance: choose the cluster based on the tree that minimizes cluster variance at a particular split.
    # splits in dendogram: If I want 2 clusters, I would cluster A, (B, C, D, E, F, G). If I wanted 4 clusters, I would have A, (B, C, D),
        # (E, F), G. Tne clusters are based on splits in a tree diagram.

# single linkage is the distance between the closest pairt of points in a cluster.

# suppose we have a blue and a red cluster and want to compute the similarity between them since we might want to consider merging them into
    # a single cluster. One metric we can use is called single linkage. So we find the 2 points that are closest to each other in this cluster.

# complete linkage is the distance betweeen the farthest pair. It is the opposite of single linkage

# average linkage is the averaged distance between all pairs. This is the middle ground between single and complete linkage. We take all pairs
    # of red and blue points, sum them up and take the average.

# below are some properties of the linkages
    # single linkage: may produce chaining i.e. sequence of close/similar clusters grouped early on
    # complete linkage: we may not merge together close groups because of outliers.
    # average linkage: compromise between single and complete – depends on the closeness/similarities being on the same scale.

# another linkage example Ward’s linkage which is based on ANOVA and is more complicated and we will not be discussing it here.

# linkage metrics are decided based on the type of data that we have

# advantages to HAC
    # simple algorithm, easy to implement. Start with each point in its own cluster, keep merging them until we get 1 cluster.
    # creates a human-interpretable structure (dendogram) for cluster grouping.

# disadvantages to HAC 
    # susceptible to noise and outliers (DBCAN has no such notion).
    # cluster groupings early drastically affect the final grouping.
    # forces a hierarchical structure on data that might not be hierarchical.

################################

import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# datsets on which algorithms will be tried 
from datasets import (
    circles, 
    moons,
    blobs,
    anisotropic,
    random,
    varied_variances,
)

# trying circles dataset
X = circles()

# HAC algorithm with ward linkage (default linkage) specifying 2 clusters
hac = AgglomerativeClustering(n_clusters=2)
hac.fit(X)

# creating scatter plot
plt.scatter(X[:,0], X[:,1], c=hac.labels_) #all the rows only first col, all the rows, 2nd col.
plt.show()

# not working to well probably because we are using ward linkage. Ward linkages groups points so that the overall variance in minimized

# trying again using single linkage metric

# HAC algorithm single linkage specifying 2 clusters
hac = AgglomerativeClustering(n_clusters=2, linkage="single")
hac.fit(X)
# plot
plt.scatter(X[:,0], X[:,1], c=hac.labels_) #all the rows only first col, all the rows, 2nd col.
plt.show()

# trying moons dataset
X = moons()

# HAC algorithm single linkage specifying 2 clusters
hac = AgglomerativeClustering(n_clusters=2, linkage="single")
hac.fit(X)
# plot
plt.scatter(X[:,0], X[:,1], c=hac.labels_) #all the rows only first col, all the rows, 2nd col.
plt.show()

# worked perfectly testing moons dataset with ward linkage

# HAC algorithm with ward linkage (default linkage) specifying 2 clusters
hac = AgglomerativeClustering(n_clusters=2)
hac.fit(X)

# creating scatter plot
plt.scatter(X[:,0], X[:,1], c=hac.labels_) #all the rows only first col, all the rows, 2nd col.
plt.show()

# trying blobs dataset 
X = blobs()

# HAC algorithm with ward linkage (default linkage) specifying 3 clusters
hac = AgglomerativeClustering(n_clusters=3)
hac.fit(X)

# creating scatter plot
plt.scatter(X[:,0], X[:,1], c=hac.labels_) #all the rows only first col, all the rows, 2nd col.
plt.show()

# HAC algorithm with specifying 3 clusters single linkage
hac = AgglomerativeClustering(n_clusters=3, linkage = "single")
hac.fit(X)

# creating scatter plot
plt.scatter(X[:,0], X[:,1], c=hac.labels_) #all the rows only first col, all the rows, 2nd col.
plt.show()

# both single and ward linkages do quite well with the blobs data set. Single works well here as there is a good distance between the blobs,
#    wards works well as the variance between these blobs is the same.

# trying anisotropic dataset 
X = anisotropic()

# HAC algorithm with specifying 3 clusters single linkage
hac = AgglomerativeClustering(n_clusters=3, linkage = "single")
hac.fit(X)

# creating scatter plot
plt.scatter(X[:,0], X[:,1], c=hac.labels_) #all the rows only first col, all the rows, 2nd col.
plt.show()

# we see a clear distinction between the 3 clusters. The data is fairly blobby.

# testing complete linkage with the above data. Three clusters
hac = AgglomerativeClustering(n_clusters=3, linkage='complete')
hac.fit(X)

# creating scatter plot
plt.scatter(X[:,0], X[:,1], c=hac.labels_) #all the rows only first col, all the rows, 2nd col.
plt.show()

# testing average linkage with above data 3 clusters
hac = AgglomerativeClustering(n_clusters=3, linkage='average')
hac.fit(X)

# creating scatter plot
plt.scatter(X[:,0], X[:,1], c=hac.labels_) #all the rows only first col, all the rows, 2nd col.
plt.show()

# through testing all three linkage option we can see average linkage is the middle ground between single and complete linkage and also
#    tends to work out. Complete linkage did not work very well and skews the clustering towards the single linkage style.

# testing algorithm on random dataset 

X = random()

# testing single linkage with above data 3 clusters
hac = AgglomerativeClustering(n_clusters=3, linkage='single') 
hac.fit(X)

# creating scatter plot
plt.scatter(X[:,0], X[:,1], c=hac.labels_) #all the rows only first col, all the rows, 2nd col.
plt.show()

# testing ward linkage random data three clusters ward linkage
hac = AgglomerativeClustering(n_clusters=3, linkage='ward')
hac.fit(X)

# creating scatter plot
plt.scatter(X[:,0], X[:,1], c=hac.labels_) #all the rows only first col, all the rows, 2nd col.
plt.show()

# testing algorithm on varied variance dataset
X = varied_variances()

# testing single linkage with above data 3 clusters
hac = AgglomerativeClustering(n_clusters=3, linkage='single') 
hac.fit(X)

# creating scatter plot
plt.scatter(X[:,0], X[:,1], c=hac.labels_) #all the rows only first col, all the rows, 2nd col.
plt.show()

# not a great fit testing ward linkage

# testing ward linkage random data three clusters ward linkage
hac = AgglomerativeClustering(n_clusters=3, linkage='ward')
hac.fit(X)

# creating scatter plot
plt.scatter(X[:,0], X[:,1], c=hac.labels_) #all the rows only first col, all the rows, 2nd col.
plt.show()

# the clustering correctly works here. This is because ward looks at the variances. The points in each cluster above, have similar variance.
#    If you remember, DBSCAN did not work too well on this data.

# we saw that with Agglomerative Clustering, changing the linkage type can drastically change the results of clustering. With blob data sets
#    with a particular shape, single linkage is probably the way to go. With data sets where the blobs have different variances, we need to
#    use something like ward. Feel free to experiment with the other kinds of linkage metrics that we did not touch on.

