# [Kmeans Clustering Scala](https://github.com/werowe/InsightML/blob/master/Kmeans.scala)

This Scala code is a self-contained application using Apache Ignite for distributed computing and Ignite Machine Learning (ML) framework for KMeans clustering. Here's a detailed explanation of what it does:

---

### Overview

- The code reads a CSV dataset containing labeled feature vectors.
- It loads the data into an Ignite distributed cache.
- It uses Ignite ML's KMeans clustering algorithm to train a model with 2 clusters.
- It retrieves and prints the cluster centers.
- It then iterates over the cached data points, predicts cluster assignments for each, and prints them.


[Java Code](https://github.com/werowe/InsightML/blob/master/LRExample.java)

