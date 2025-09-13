Project 1: Basic Cluster Simulation

 Description
In this project, a simple function is used to create synthetic 2D data (Income and Age) with multiple clusters. The K-Means algorithm from `scikit-learn` is then applied to identify the clusters, and the results are visualized using `matplotlib`.

 Features
- Generates synthetic data with `N` points and `K` clusters
- Uses random normal distributions around cluster centroids
- Standardizes data before applying K-Means
- Visualizes clustered points using scatter plots
- Shows which cluster each point belongs to using color coding

Project 2: Advanced K-Means Clustering with Analysis
Description

This project provides a more advanced simulation of customer data with predefined cluster parameters. It demonstrates a full pipeline including data standardization, K-Means clustering, visualization of clusters with centroids, and basic cluster statistics.

Features
Generates multiple clusters with realistic parameters for Income and Age
Standardizes data using StandardScaler
Applies K-Means with optimized initialization (k-means++) and multiple runs (n_init=20) for stability
Visualizes clusters and centroids in a 2D scatter plot
Calculates and prints mean Income, mean Age, and number of points per cluster

Requirements :
Python 3.x
NumPy
scikit-learn
matplotlib

Learning Outcomes :
Understanding K-Means clustering conceptually and mathematically
Creating synthetic datasets for clustering exercises
Visualizing clusters and centroids
Analyzing cluster statistics such as mean values and cluster sizes
Practicing standardization of features for clustering