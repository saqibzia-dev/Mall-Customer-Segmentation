import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values
# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler"""
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train.reshape(-1,1))
"""Using elbow method to find number of optimal clusters"""
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++",n_init = 10,max_iter = 100)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('No. of clusters')
plt.ylabel('wcss')
plt.figure("K-Mean Clustering",clear = True)

kmeans = KMeans(n_clusters=5,init="k-means++",n_init = 10,max_iter = 300,random_state = 42)
y_kmeans = kmeans.fit_predict(X)
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s = 100,c="red",label="low income high spend")
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s = 100,c="green",label="in middle(Target)")
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s = 100,c="blue",label="high income high spend")
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s = 100,c = "cyan",label="low income low spend")
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s = 100,c = "magenta",label="high income low spend")
print(kmeans.cluster_centers_)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='black',label='Centroids')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.title('Clusters of Customers')
plt.legend()
plt.show()