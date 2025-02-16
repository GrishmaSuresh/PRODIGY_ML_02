import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def find_optimal_clusters(data, max_k=10):
    """Using the Elbow Method to determine the optimal number of clusters."""
    inertia = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k+1), inertia, marker='o', linestyle='--')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.show()

def apply_kmeans(data, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans

if __name__ == "__main__":
    from preprocess import load_data, preprocess_data
    df = load_data("../data/Mall_Customers.csv")
    data = preprocess_data(df)

    find_optimal_clusters(data)
    clusters, model = apply_kmeans(data, n_clusters=5)
    print("Clustering Completed!")
