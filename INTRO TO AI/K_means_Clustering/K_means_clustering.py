import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load credit card customer dataset from Kaggle by Arya Shah
file_path = '/content/sample_data/Credit_Card_Customer_Data.csv'
data = pd.read_csv(file_path)

# Select relevant features for clustering and drop rows with missing values
X = data[['Total_visits_online', 'Total_calls_made']].dropna().values

# Elbow method to find the optimal number of clusters
sse = []  # Sum of squared error (inertia)
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=2)
    km.fit(X)
    sse.append(km.inertia_)

# Plotting the elbow method
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))
g = sns.lineplot(x=range(1, 11), y=sse)
g.set(xlabel="Number of Clusters (k)", ylabel="Sum Squared Error", title="Elbow Method for Optimal Clusters")
plt.show()

# Set number of clusters based on elbow method (choose k from the plot)
# Step 1: Specify the number k of clusters
optimal_k = 3  # Set this value based on the elbow plot

# Define the k-means algorithm function (custom implementation)
def k_means(X, k, max_iters=100):
    # Step 2: Randomly initialize k centroids
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        # Step 3: Repeat
        # Step 4: Expectation step - Assign each point to its closest centroid
        clusters = [[] for _ in range(k)]
        for x in X:
            distances = [np.linalg.norm(x - centroid) for centroid in centroids]
            closest_centroid = np.argmin(distances)
            clusters[closest_centroid].append(x)
        
        # Step 5: Maximization step - Compute the new centroid (mean) of each cluster
        new_centroids = np.array([np.mean(cluster, axis=0) if cluster else centroids[i]
                                  for i, cluster in enumerate(clusters)])
        
        # Step 6: Until the centroid positions do not change
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return centroids, clusters

# Run k-means algorithm on the customer data with the optimal k
centroids, clusters = k_means(X, optimal_k)

# Plotting the results (using Aboard and Fatalities for visualization)
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
for i, cluster in enumerate(clusters):
    cluster = np.array(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i], label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=100, label='Centroids')
plt.xlabel('Total_visits_online')
plt.ylabel('Total_calls_made')
plt.legend()
plt.title(f'K-means Clustering on Airplane Crashes and Fatalities (k={optimal_k})')
plt.show()
