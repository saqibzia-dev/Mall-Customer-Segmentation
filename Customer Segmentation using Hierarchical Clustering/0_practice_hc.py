import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#finding optimal clusters for our data
import scipy.cluster.hierarchy as sch
dendo = sch.dendrogram(sch.linkage(X,method = 'ward'))
plt.xlabel('customers')
plt.ylabel('euclidean distance')
plt.title('dendrogram')
plt.figure("Hierarchichal Clustering",clear = True)
#fitting optimal number of hierarchichal clusterers on data and plotting it
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")
y_hc = hc.fit_predict(X)
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c="red",label = "cluster1")
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.legend()
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title('Hierarchichal Clustering')
plt.show()