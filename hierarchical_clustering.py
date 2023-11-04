-- here also we are using Mall_Customers.csv file

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')

dataset.head()

"""Our goal is to cluster the customers based on their Spending score. That’s why CustomerID and Genre are useless. So we remove both columns."""

X = dataset.iloc[:, [3, 4]].values

"""Now, we have only Annual Income and Spending Score Column."""

X

"""We have loaded dataset. Now its time to find the optimal number of clusters. And for that we need to create a Dendrogram."""

# Create Dendrogram to find the Optimal Number of Clusters

import scipy.cluster.hierarchy as sch
dendro = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

"""Here in the code “sch” is the short code for scipy.cluster.hierarchy.”

“dendro” is the variable name. It may be anything. And “Dendrogram” is the function name.

So, after implementing this code, we will get our Dendrogram.

As I discussed that cut the horizontal line with longest line that traverses maximum distance up and down without intersecting the merging points.

In that dendrogram, the optimal number of clusters are 5.

Now let’s fit our Agglomerative model with 5 clusters.
"""

# Fitting Agglomerative Hierarchical Clustering to the dataset

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

"""Now our model has been trained. If you want to see different clusters, you can do it by simply writing print.


"""

print(y_hc)

"""Now, its time to visualize the clusters.

"""

# Visualise the clusters


plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

