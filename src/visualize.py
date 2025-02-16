import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_clusters(df, labels):
    """Visualize clusters with a scatter plot."""
    df['Cluster'] = labels
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['Cluster'], palette="viridis", s=100)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Customer Segmentation using K-Means")
    plt.show()

if __name__ == "__main__":
    from preprocess import load_data, preprocess_data
    from clustering import apply_kmeans

    df = load_data("../data/Mall_Customers.csv")
    data = preprocess_data(df)

    clusters, _ = apply_kmeans(data, n_clusters=5)
    plot_clusters(df, clusters)
