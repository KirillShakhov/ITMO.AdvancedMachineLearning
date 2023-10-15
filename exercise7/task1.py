import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml

# Fetch the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)

# Prepare the data
X = np.array(mnist.data.astype('float64'))
y = np.array(mnist.target.astype('int64'))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Initialize and train the KMeans model
kmeans_model = KMeans(n_clusters=10, random_state=21, n_init=10, max_iter=300)
kmeans_model.fit(X_train)

# Predict the labels of the training data
predicted_labels = kmeans_model.labels_

# Find the number of elements assigned to the cluster with label 9
num_elements_cluster_9 = np.sum(predicted_labels == 9)

print(f"Number of elements in cluster labeled '9': {num_elements_cluster_9}")

import matplotlib.pyplot as plt

# Get the centroids of the clusters
centroids = kmeans_model.cluster_centers_

# Set up the figure
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.ravel()

# Loop through each centroid and plot it as an image
for i, centroid in enumerate(centroids):
    axes[i].imshow(centroid.reshape(28, 28), cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f'Cluster {i}')

plt.tight_layout()
plt.show()



# Importing necessary libraries and re-executing the steps

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml

# Fetch the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)

# Prepare the data
X = np.array(mnist.data.astype('float64'))
y = np.array(mnist.target.astype('int64'))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Initialize and train the KMeans model
kmeans_model = KMeans(n_clusters=10, random_state=21, n_init=10, max_iter=300)
kmeans_model.fit(X_train)

# Get the centroids of the clusters
centroids = kmeans_model.cluster_centers_

# Set up the figure
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.ravel()

# Loop through each centroid and plot it as an image
for i, centroid in enumerate(centroids):
    axes[i].imshow(centroid.reshape(28, 28), cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f'Cluster {i}')

plt.tight_layout()
plt.show()
