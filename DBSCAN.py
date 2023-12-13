# DBSCAN (Density based Spatial Clustering of Applications with Noise). It takes a density based approach. It groups together points in
#    high-density regions and ignores outliers/noise in low-density regions. DBSCAN has the notion of noise points i.e. points that don’t 
#    belong in a particular cluster. If you are performing clustering on any noisy data, DBSCAN should be on the top of your list of 
#    clustering algorithms. 

# the algorithm takes two parameters
    # ε – size of the neighborhood
    # minPts – the density requirement of the neighborhood

# ε neighborhood is the set of all point at most ε away from p
#  minPts is a parameter to denote high density.If there are at least minPts in the ε neighborhood, then this is a high-density region

# there is no parameter for number of clusters because it is inferred from the data. Points in high-density region are defined as being in 1
#   cluster. Each point is labelled according to the region it is in.bA core point is one that belongs to a cluster, a border point kind of
#   belongs to a cluster but lie outside the main high-density region. Finally the outliers/noise points lie outside the cluster.

# p is a core point if it has at least minPts in its ε neighborhood. 

# by changing ε and minPts, we can change the density parameter. If the data is sparse, we may want to adjust these parameters correctly so
#   that we can still correctly label clusters. Likewise if the data has very dense clusters, we may want to adjust the parameters accordingly.

# q is a border point it if is reachable from some core point p.

# the definition of reachable is r is said to be directly reachable/density-reachable from p if r is in the ε-neighborhood of p and p is a
#    core point. a point r is directly reachable from p it satisfies this criteria. In the example above q is directly reachable from p, but
#    it not a core point as it does not have enough minPts to give that designation. A point t is reachable/ density reachable from p if there
#    exists some sequence of core points connecting p to t through their ε-neighborhoods.

# outliers are points that are not reachable from core points. 

###########################
# Pseudo code DBSCAN
###########################

# pick a point p that hasn’t been selected or labeled yet.

# check the number of points in p’s ε-neighborhood

    # if it is less than minPts, mark p as an outlier for now and go back to first step

    # if it is at least minPts, mark p as a core point and start a new cluster at p

# now that we have point p, find all reachable points from p
    
    # mark some point q as a core point if q has at least minPts in its ε-neighborhood
    
    # mark some point q as a border point if q does not have at least minPts in its ε-neighborhood but is reachable from p

# go back to first step and repeat until each point is labeled (core / border / outlier)

######################
###################### 

# advantages to DBSCAN
    # robust to noise and outliers because of how we select ε and minPts.
    # number of clusters in inferred from the data.
    # correctly groups arbitrary cluster shapes (circular / elliptical etc).

# disadvantages to DBSCAN
    # very sensitive to parameters. Changes to ε or minPts can produce varying clusters.
    # unable to handle varying densities. If one set of data has very dense clusters and sparse clusters, DBSCAN does not handle this too well.
        # this is because we have the same density parameters for all the points. Varying these parameters is quite difficult to do.
    # the quality of DBSCAN is dependent on which distance metric we use. We generally use Euclidean distance which does not perform well on
        # data of higher dimensions (curse of dimensionality). For our purposes we will be looking only at 2D data.

################################

# import matplotlib and sklearn cluster
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# datsets on which algorithms will be tried 

from datasets import (
    circles, 
    moons,
    blobs,
    anisotropic,
    random,
    varied_variances,
)

X = circles()

# clustering algorithm eps value 0.1 min samples value 5
dbscan = DBSCAN(eps=0.1, min_samples=5)
dbscan.fit(X)

# get inliers (label not equal to -1) and their cluster
X_inlier = X[dbscan.labels_ != -1]

# color of cluster such that they are not outliers.
y_inlier  = dbscan.labels_[dbscan.labels_ != -1]

# get outliers
X_outlier = X[dbscan.labels_ == -1]

# changing colors of scatter plot
plt.scatter(X_inlier[:,0], X_inlier[:,1], c=y_inlier, cmap='viridis')

# outliers – color black
plt.scatter(X_outlier[:,0], X_outlier[:,1], c='red')
plt.show()

# changing eps value to be larger

# clustering algorithm eps value 0.5 min samples value 5
dbscan = DBSCAN(eps=0.5, min_samples=5)

dbscan.fit(X)

# get inliers (label not equal to -1) and their cluster
X_inlier = X[dbscan.labels_ != -1]

# color of cluster such that they are not outliers.
y_inlier  = dbscan.labels_[dbscan.labels_ != -1]

# get outliers
X_outlier = X[dbscan.labels_ == -1]

# plot with appropriate colors (color of cluster they belong to)
plt.scatter(X_inlier[:,0], X_inlier[:,1], c=y_inlier, cmap= "viridis")

#note the algorithm was clearly able to recognize the two distinct clusters

# outliers – color black
plt.scatter(X_outlier[:,0], X_outlier[:,1], c='red')

# display plot
plt.show()

# notice that everything is being denoted as being part of 1 cluster and that there are no outliers. DBSCAN is skipping across the 2 regions
#    probably using the points in between as a bridge. This value, 0.5, is clearly too high

# running DBSCAN on moons data set

X = moons()

# clustering algorithm eps value 0.1 min samples value 5
dbscan = DBSCAN(eps=0.1, min_samples=5)
dbscan.fit(X)

# get inliers (label not equal to -1) and their cluster
X_inlier = X[dbscan.labels_ != -1]

# color of cluster such that they are not outliers.
y_inlier  = dbscan.labels_[dbscan.labels_ != -1]

# get outliers
X_outlier = X[dbscan.labels_ == -1]

# changing colors of scatter plot
plt.scatter(X_inlier[:,0], X_inlier[:,1], c=y_inlier, cmap='viridis')

# outliers – color black
plt.scatter(X_outlier[:,0], X_outlier[:,1], c='red')
plt.show()

# note the two outliers and that the algorithm was able to recognize the two distinct clusters

X = blobs()

# clustering algorithm eps value 0.1 min samples value 5
dbscan = DBSCAN(eps=0.1, min_samples=5)
dbscan.fit(X)

# get inliers (label not equal to -1) and their cluster
X_inlier = X[dbscan.labels_ != -1]

# color of cluster such that they are not outliers.
y_inlier  = dbscan.labels_[dbscan.labels_ != -1]

# get outliers
X_outlier = X[dbscan.labels_ == -1]

# changing colors of scatter plot
plt.scatter(X_inlier[:,0], X_inlier[:,1], c=y_inlier, cmap='viridis')

# outliers – color black
plt.scatter(X_outlier[:,0], X_outlier[:,1], c='red')
plt.show()

# all points are being colored black meaning they are being recognized as outliers. This indicates that our eps value is likely too small

# clustering algorithm eps value 0.5 min samples value 5
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)

# get inliers (label not equal to -1) and their cluster
X_inlier = X[dbscan.labels_ != -1]

# color of cluster such that they are not outliers.
y_inlier  = dbscan.labels_[dbscan.labels_ != -1]

# get outliers
X_outlier = X[dbscan.labels_ == -1]

# changing colors of scatter plot
plt.scatter(X_inlier[:,0], X_inlier[:,1], c=y_inlier, cmap='viridis')

# outliers – color black
plt.scatter(X_outlier[:,0], X_outlier[:,1], c='red')
plt.show()

# three distinct clusters appears however many of the outlier points look close enough to be a part of the cluster so we will increase eps
#   value one more time

# clustering algorithm eps value 0.75 min samples value 5
dbscan = DBSCAN(eps=0.75, min_samples=5)
dbscan.fit(X)

# get inliers (label not equal to -1) and their cluster
X_inlier = X[dbscan.labels_ != -1]

# color of cluster such that they are not outliers.
y_inlier  = dbscan.labels_[dbscan.labels_ != -1]

# get outliers
X_outlier = X[dbscan.labels_ == -1]

# changing colors of scatter plot
plt.scatter(X_inlier[:,0], X_inlier[:,1], c=y_inlier, cmap='viridis')

# outliers – color black
plt.scatter(X_outlier[:,0], X_outlier[:,1], c='red')
plt.show()

# trying algorithm on anisotropic data
X = anisotropic()

# clustering algorithm eps value 0.75 min samples value 5
dbscan = DBSCAN(eps=0.75, min_samples=5)
dbscan.fit(X)

# get inliers (label not equal to -1) and their cluster
X_inlier = X[dbscan.labels_ != -1]

# color of cluster such that they are not outliers.
y_inlier  = dbscan.labels_[dbscan.labels_ != -1]

# get outliers
X_outlier = X[dbscan.labels_ == -1]

# changing colors of scatter plot
plt.scatter(X_inlier[:,0], X_inlier[:,1], c=y_inlier, cmap='viridis')

# outliers – color black
plt.scatter(X_outlier[:,0], X_outlier[:,1], c='red')
plt.show()

# we get 3 clusters with a couple of outliers. These outliers are quite far away from any of the clusters. we will lower the eps value

# clustering algorithm eps value 0.5 min samples value 5
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)

# get inliers (label not equal to -1) and their cluster
X_inlier = X[dbscan.labels_ != -1]

# color of cluster such that they are not outliers.
y_inlier  = dbscan.labels_[dbscan.labels_ != -1]

# get outliers
X_outlier = X[dbscan.labels_ == -1]

# changing colors of scatter plot
plt.scatter(X_inlier[:,0], X_inlier[:,1], c=y_inlier, cmap='viridis')

# outliers – color black
plt.scatter(X_outlier[:,0], X_outlier[:,1], c='red')
plt.show()

# trying algorithm on random data set
X = random()

# clustering algorithm eps value 0.5 min samples value 5
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)

# get inliers (label not equal to -1) and their cluster
X_inlier = X[dbscan.labels_ != -1]

# color of cluster such that they are not outliers.
y_inlier  = dbscan.labels_[dbscan.labels_ != -1]

# get outliers
X_outlier = X[dbscan.labels_ == -1]

# changing colors of scatter plot
plt.scatter(X_inlier[:,0], X_inlier[:,1], c=y_inlier, cmap='viridis')

# outliers – color black
plt.scatter(X_outlier[:,0], X_outlier[:,1], c='red')
plt.show()

# everything has been put into one cluster probably because the eps value is set too high 

# clustering algorithm eps value 0.1 min samples value 5
dbscan = DBSCAN(eps=0.1, min_samples=5)
dbscan.fit(X)

# get inliers (label not equal to -1) and their cluster
X_inlier = X[dbscan.labels_ != -1]

# color of cluster such that they are not outliers.
y_inlier  = dbscan.labels_[dbscan.labels_ != -1]

# get outliers
X_outlier = X[dbscan.labels_ == -1]

# changing colors of scatter plot
plt.scatter(X_inlier[:,0], X_inlier[:,1], c=y_inlier, cmap='viridis')

# outliers – color black
plt.scatter(X_outlier[:,0], X_outlier[:,1], c='red')
plt.show()

# still only one cluster trying again with lower eps value 

# clustering algorithm eps value 0.01 min samples value 5
dbscan = DBSCAN(eps=0.01, min_samples=5)
dbscan.fit(X)

# get inliers (label not equal to -1) and their cluster
X_inlier = X[dbscan.labels_ != -1]

# color of cluster such that they are not outliers.
y_inlier  = dbscan.labels_[dbscan.labels_ != -1]

# get outliers
X_outlier = X[dbscan.labels_ == -1]

# changing colors of scatter plot
plt.scatter(X_inlier[:,0], X_inlier[:,1], c=y_inlier, cmap='viridis')

# outliers – color black
plt.scatter(X_outlier[:,0], X_outlier[:,1], c='red')
plt.show()

# now most of the random data set is being classified as outlier. This is not a good algorithm for this data set 

# trying algorithm on varied variances data set 
X = varied_variances()

# clustering algorithm eps value 0.1 min samples value 5
dbscan = DBSCAN(eps=0.1, min_samples=5)
dbscan.fit(X)

# get inliers (label not equal to -1) and their cluster
X_inlier = X[dbscan.labels_ != -1]

# color of cluster such that they are not outliers.
y_inlier  = dbscan.labels_[dbscan.labels_ != -1]

# get outliers
X_outlier = X[dbscan.labels_ == -1]

# changing colors of scatter plot
plt.scatter(X_inlier[:,0], X_inlier[:,1], c=y_inlier, cmap='viridis')

# outliers – color black
plt.scatter(X_outlier[:,0], X_outlier[:,1], c='red')
plt.show()

# we see many many outlier points indicating that our eps value is probably too low

# clustering algorithm eps value 1 min samples value 5
dbscan = DBSCAN(eps=1, min_samples=5)
dbscan.fit(X)

# get inliers (label not equal to -1) and their cluster
X_inlier = X[dbscan.labels_ != -1]

# color of cluster such that they are not outliers.
y_inlier  = dbscan.labels_[dbscan.labels_ != -1]

# get outliers
X_outlier = X[dbscan.labels_ == -1]

# changing colors of scatter plot
plt.scatter(X_inlier[:,0], X_inlier[:,1], c=y_inlier, cmap='viridis')

# outliers – color black
plt.scatter(X_outlier[:,0], X_outlier[:,1], c='red')
plt.show()

# interestingly DBSCAN does not work very well with this kind of data. It works well when clusters have about the same kind of variance. This
#    is because the parameters eps and min_samples are universal for all our data. We don’t adjust it for each cluster.

# some of the data points marked as outliers are actually a part of the cluster so let us try to increase the eps value again 

# clustering algorithm eps value 2 min samples value 5
dbscan = DBSCAN(eps=2, min_samples=5)
dbscan.fit(X)

# get inliers (label not equal to -1) and their cluster
X_inlier = X[dbscan.labels_ != -1]

# color of cluster such that they are not outliers.
y_inlier  = dbscan.labels_[dbscan.labels_ != -1]

# get outliers
X_outlier = X[dbscan.labels_ == -1]

# changing colors of scatter plot
plt.scatter(X_inlier[:,0], X_inlier[:,1], c=y_inlier, cmap='viridis')

# outliers – color black
plt.scatter(X_outlier[:,0], X_outlier[:,1], c='red')
plt.show()

# in this case the eps value was too large and combined the two groups into one cluster 

# note that the eps value controls for things like outliers

# now consider the following examples where we vary the value of min samples

X = blobs()

# clustering algorithm eps value 1 min samples value 5
dbscan = DBSCAN(eps=1, min_samples=5)
dbscan.fit(X)

# get inliers (label not equal to -1) and their cluster
X_inlier = X[dbscan.labels_ != -1]

# color of cluster such that they are not outliers.
y_inlier  = dbscan.labels_[dbscan.labels_ != -1]

# get outliers
X_outlier = X[dbscan.labels_ == -1]

# changing colors of scatter plot
plt.scatter(X_inlier[:,0], X_inlier[:,1], c=y_inlier, cmap='viridis')

# outliers – color black
plt.scatter(X_outlier[:,0], X_outlier[:,1], c='red')
plt.show()

# shows pretty good clustering. Now showing what happens when alter min sample 

# clustering algorithm eps value 0.1 min samples value 2
dbscan = DBSCAN(eps=1, min_samples=2)
dbscan.fit(X)

# get inliers (label not equal to -1) and their cluster
X_inlier = X[dbscan.labels_ != -1]

# color of cluster such that they are not outliers.
y_inlier  = dbscan.labels_[dbscan.labels_ != -1]

# get outliers
X_outlier = X[dbscan.labels_ == -1]

# changing colors of scatter plot
plt.scatter(X_inlier[:,0], X_inlier[:,1], c=y_inlier, cmap='viridis')

# outliers – color black
plt.scatter(X_outlier[:,0], X_outlier[:,1], c='red')
plt.show()

# Small clusters are created at the bottom that aren’t really clusters revealing that a low min_samples value creates more clusters. 

# clustering algorithm eps value 1 min samples value 20
dbscan = DBSCAN(eps=1, min_samples=20)
dbscan.fit(X)

# get inliers (label not equal to -1) and their cluster
X_inlier = X[dbscan.labels_ != -1]

# color of cluster such that they are not outliers.
y_inlier  = dbscan.labels_[dbscan.labels_ != -1]

# get outliers
X_outlier = X[dbscan.labels_ == -1]

# changing colors of scatter plot
plt.scatter(X_inlier[:,0], X_inlier[:,1], c=y_inlier, cmap='viridis')

# outliers – color black
plt.scatter(X_outlier[:,0], X_outlier[:,1], c='red')
plt.show()

# with a high min sample value we create more outlier points 

# let us try varied variances data set with a high min samples value 

# trying algorithm on varied variances data set 
X = varied_variances()

# clustering algorithm eps value 1 min samples value 20
dbscan = DBSCAN(eps=1, min_samples=20)
dbscan.fit(X)

# get inliers (label not equal to -1) and their cluster
X_inlier = X[dbscan.labels_ != -1]

# color of cluster such that they are not outliers.
y_inlier  = dbscan.labels_[dbscan.labels_ != -1]

# get outliers
X_outlier = X[dbscan.labels_ == -1]

# changing colors of scatter plot
plt.scatter(X_inlier[:,0], X_inlier[:,1], c=y_inlier, cmap='viridis')

# outliers – color black
plt.scatter(X_outlier[:,0], X_outlier[:,1], c='red')
plt.show()

# we get a ton of outlier points around the green region. This is because this region does not have enough points to satisfy the core points
#   criteria as the points are quite spread out.

# DBSCAN works well for all the clusters that have different shapes, but the catch is that they need to have about the same variance. It does
#    not work well for clusters that have different variances or spreads.

