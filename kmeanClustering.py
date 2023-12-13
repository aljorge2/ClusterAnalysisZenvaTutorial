# kmeans is a popular easy to implement clustering algorithm. The algorithm separates the data into k disjoin clusters. These clusters are
#    defined such that they minimize the within-cluster sum-of-squares. Disjoint here means that 1 point cannot belong to more than 1 cluster.
#    There is only 1 parameter for this algorithm i.e. k (the number of clusters). We must know k before we run the algorithm.

###########################
# Pseudo code Kmeans
###########################

# randomly initialize the cluster centers. For example we can pick 2 random points to initialize the clusters.

# assign each point to it’s nearest cluster using distance formula like Euclidian distance.

# update the cluster centroids using the mean of the points assigned to it.

# go back to second step until convergence (the cluster centroids stop moving or they move small imperceptible amounts).

######################
###################### 


# k-means is sensitive to where you initialize the centroids. There are a few techniques to do this:
    # assign each cluster center to a random data point.
    # choose k points to be farthest away from each other within the bounds of the data.
    # repeat k-means over and over again and pick the average of the clusters.
    # another advanced approach called k-means ++ does things like ANOVA (Analysis Of Variance). We won’t be getting into it though.

# one way to choose k is called the elbow method. The steps for the elbow method are as follows
    # choose some values of k and run the clustering algorithm 
    # for each cluster, compute the within-cluster sum-of-squares between the centroid and each data point.
    # sum up for all clusters, plot on a graph
    # repeat for different values of k, keep plotting on the graph.
    # then pick the elbow of the graph.

# advantages Of k-means
    # this is widely known and used algorithm.
    # it’s also fairly simple to understand and easy to implement.
    # it is also guaranteed to converge.

# disadvantages of k-means
    # it is algorithmically slow i.e. can take a long time to converge.
    # it may also not converge to the local minima i.e. the optimal solution.
    # it’s also not very robust against varying cluster shapes e.g. It may not perform very well for elongated cluster shapes. 
        # this is because we use the same parameters for each cluster.

################################

# import matplotlib and sklearn cluster
from matplotlib.cm import _colormaps
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

# datsets on which algorithms will be tried 

from datasets import (
    circles, 
    moons,
    blobs,
    anisotropic,
    random,
    varied_variances,
)
# circles
X=circles()

# making scatter plot matplotlib
plt.figure()
plt.scatter(X[:,0], X[:,1])  #all the rows only first col, all the rows, 2nd col.
plt.show()

# based on plot below we can guess there are two clusters

# random_state helps makes this deterministic.
kmeans = KMeans(n_clusters=2, random_state=17) 
kmeans.fit(X)

# add new figure prior to plt.show()
plt.figure()

# kmeans class labels_ is a vector that assigns each point to a numerical cluster 
# matplotlib converts that to an actual color
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_)
plt.show()

# what k-means came up with is not the ideal clustering for k-means. Working with concentric data does not work out too well. 
#   k-means is unable to handle different kinds of clustering shapes – in this case a circle is something it can’t handle very well.

# trying k-means on a moon dataset
X = moons()

#plotting moon data set via scatterplot using all the rows only first col, all the rows, 2nd col.
plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.show()

#kmeans with 2 clusters using random state to make it deterministic
kmeans  = KMeans(n_clusters=2, random_state=17,)
kmeans.fit(X)
plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.figure()
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_,cmap="viridis")
plt.show()

# trying k-mans on a blobs data set

X= blobs()

kmeans = KMeans(n_clusters=3, random_state=17)
kmeans.fit(X)
plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.figure()
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_)
plt.show()

# trying k-means on a anisotropic dataset 
X = anisotropic()

#kmeans with 3 clusters using random state to make it deterministic
kmeans  = KMeans(n_clusters=3, random_state=17)
kmeans.fit(X)
plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.figure()
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_)
plt.show()

# trying k means on random data set 
X = random()

#kmeans with 3 clusters using random state to make it deterministic
kmeans  = KMeans(n_clusters=3, random_state=17)
kmeans.fit(X)
plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.figure()
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_)
plt.show()

#trying k means on varied variances data set
X = varied_variances()

#kmeans with 3 clusters using random state to make it deterministic
kmeans  = KMeans(n_clusters=3, random_state=17)
kmeans.fit(X)
plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.figure()
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_)
plt.show()