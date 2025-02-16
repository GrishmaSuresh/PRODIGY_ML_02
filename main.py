from src.preprocess import load_data, preprocess_data
from src.clustering import apply_kmeans
from src.visualize import plot_clusters

# Load and preprocess data
df = load_data("data/Mall_Customers.csv")
data = preprocess_data(df)

# Apply K-Means
clusters, model = apply_kmeans(data, n_clusters=5)

# Visualize clusters
plot_clusters(df, clusters)
