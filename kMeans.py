import numpy as np
import matplotlib.pyplot as plt

# Function to initialize centroids by randomly selecting k points from the dataset
def initialize_centroids(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

# Function to compute Euclidean distances between each data point and each centroid
def compute_distances(X, centroids):
    # X has shape (n_samples, n_features) and centroids (k, n_features)
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return distances

# Function to assign each data point to the nearest centroid
def assign_clusters(X, centroids):
    distances = compute_distances(X, centroids)
    return np.argmin(distances, axis=1)

# Function to update centroid positions based on the mean of assigned points
def update_centroids(X, labels, k):
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        points = X[labels == i]
        if len(points) > 0:
            centroids[i] = points.mean(axis=0)
        else:
            # If a cluster gets no points, reinitialize its centroid randomly
            centroids[i] = X[np.random.choice(X.shape[0])]
    return centroids

# K-Means algorithm implementation
def kmeans(X, k, max_iters=100, tol=1e-4):
    # Initialize centroids
    centroids = initialize_centroids(X, k)
    
    for i in range(max_iters):
        # Assign clusters based on current centroids
        labels = assign_clusters(X, centroids)
        
        # Update centroids using current cluster assignments
        new_centroids = update_centroids(X, labels, k)
        
        # Check for convergence (if centroids move less than tol)
        if np.all(np.abs(new_centroids - centroids) < tol):
            print(f"Converged after {i+1} iterations.")
            break
        centroids = new_centroids
        
    return labels, centroids

# ----------------------------
# Generate Synthetic Data
# ----------------------------
np.random.seed(42)
# Create three clusters of data points
X1 = np.random.randn(100, 2) + np.array([5, 5])
X2 = np.random.randn(100, 2) + np.array([-5, -5])
X3 = np.random.randn(100, 2) + np.array([5, -5])
X = np.vstack((X1, X2, X3))

# ----------------------------
# Apply K-Means Clustering
# ----------------------------
k = 3  # Number of clusters
labels, centroids = kmeans(X, k)

# ----------------------------
# Visualize the Clusters
# ----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
