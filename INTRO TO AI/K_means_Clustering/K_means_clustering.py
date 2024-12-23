import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

# Load credit card customer dataset
file_path = './Credit_Card_Customer_Data.csv'  # Update this with the correct path
data = pd.read_csv(file_path)

# Step 1: Extract relevant features for clustering
X = data.select_dtypes(include=[np.number]).values  # Select only numeric columns

# Step 2: Standardize the features (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA (reduce to 2 components for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 4: Elbow method to find the optimal number of clusters
sse = []  # Sum of squared error (inertia)
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=2)
    km.fit(X_pca)
    sse.append(km.inertia_)

# Plotting the elbow method
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))
sns.lineplot(x=range(1, 11), y=sse)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Sum Squared Error (SSE)")
plt.title("Elbow Method for Optimal Clusters")
plt.show()

# Step 5: Set number of clusters based on elbow plot
optimal_k = 3  # Adjust based on the elbow method

# Step 6: Run K-means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=2)
clusters = kmeans.fit_predict(X_pca)
centroids = kmeans.cluster_centers_

# Step 7: Evaluate performance metrics
silhouette = silhouette_score(X_pca, clusters)
davies_bouldin = davies_bouldin_score(X_pca, clusters)
# For ARI, we need ground truth labels; here we use a dummy example:
# Replace 'true_labels' with the actual ground truth column if available in your dataset
if 'true_labels' in data.columns:
    ari = adjusted_rand_score(data['true_labels'], clusters)
else:
    ari = None

# Print performance metrics
print(f"Silhouette Score: {silhouette:.3f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
if ari is not None:
    print(f"Adjusted Rand Index (ARI): {ari:.3f}")

# Step 8: Plot the results (using PCA components for visualization)
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b', 'c', 'm', 'y']
for i in range(optimal_k):
    plt.scatter(X_pca[clusters == i, 0], X_pca[clusters == i, 1],
                color=colors[i], label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1],
            color='black', marker='x', s=100, label='Centroids')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.title(f'K-means Clustering with PCA (k={optimal_k})')
plt.show()