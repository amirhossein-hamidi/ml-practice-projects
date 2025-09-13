import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

np.random.seed(42)

cluster_params = [
    (30000, 25),
    (70000, 35),
    (50000, 50),
    (100000, 40),
]

pointPerCluster = 50
X = []

for income_mean, age_mean in cluster_params:
    incomes = np.random.normal(income_mean, 8000, pointPerCluster)
    ages = np.random.normal(age_mean, 3, pointPerCluster)
    cluster_data = np.column_stack((incomes,ages))
    X.append(cluster_data)

X = np.vstack(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 4
kmeans = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42)
kmeans.fit(X_scaled)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

plt.figure(figsize=(10,6))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels, cmap='tab10', s=50, alpha=0.6)
plt.scatter(centroids[:,0], centroids[:,1], color='red', marker='X', s=200)
plt.show()

for i in range(k):
    cluster_points = X[labels == i]
    mean_income = cluster_points[:,0].mean()
    mean_age = cluster_points[:,1].mean()
    print(f'Cluster {i}: Mean Income = {mean_income:0f}, Mean Age = {mean_age:0f}, Points = {cluster_points}')