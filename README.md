# Apache Insight

## [Kmeans Clustering Scala](https://github.com/werowe/InsightML/blob/master/Kmeans.scala)

This Scala code is a self-contained application using Apache Ignite for distributed computing and Ignite Machine Learning (ML) framework for KMeans clustering. Here's a detailed explanation of what it does:

---

### Overview

- The code reads a CSV dataset containing labeled feature vectors.
- It loads the data into an Ignite distributed cache.
- It uses Ignite ML's KMeans clustering algorithm to train a model with 2 clusters.
- It retrieves and prints the cluster centers.
- It then iterates over the cached data points, predicts cluster assignments for each, and prints them.


## [Java Linear Regression Model](https://github.com/werowe/InsightML/blob/master/LRExample.java)

This Java code demonstrates training and evaluating a simple linear regression model using Apache Ignite's machine learning (ML) library. The workflow is entirely in-memory and distributed—leveraging Ignite’s caching and computational architecture.

---

### Main Steps

- **Starts an Ignite node** with a custom work directory.
- **Creates a distributed cache** to store training data.
- **Manually inserts seven labeled examples** into the cache for regression training.
- **Configures a vectorizer** to separate label from features in each vector.
- **Trains a linear regression model** using the LSQR trainer on the cache data.
- **Evaluates the model's performance** using RMSE metric.
- **Prints model properties** (intercept and weights).
- **Destroys the cache at end to clean up.**

---

### Detailed Section Breakdown

#### Ignite Node Start

