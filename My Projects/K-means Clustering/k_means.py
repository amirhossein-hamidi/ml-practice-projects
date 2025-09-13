import numpy as np
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# define a function to use k-means clustering
def create_cluster(N,K):
    np.random.seed(42)
    PointPerCluster = float(N) / K
    X = []
    for i in range(K):
        incomeCentroid = np.random.uniform(20000,200000)
        ageCentroid = np.random.uniform(20,70)
        for j in range(int(PointPerCluster)):
            X.append([np.random.normal(incomeCentroid,10000), np.random.normal(ageCentroid,2)])
    
    X = np.array(X)
    return X

# standard the data and fit with model
data = create_cluster(100,5)
model = KMeans(n_clusters=5)
model = model.fit(scale(data))
print(model.labels_) #this shows each point belongs to wich cluster

# showing the result on plot
plt.figure(figsize=(8,6))
plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(np.float32))
plt.show()