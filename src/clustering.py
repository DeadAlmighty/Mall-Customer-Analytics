# Clustering module
"""
This module contains clustering algorithms.
"""

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def elbow_method(X):
    """
    Determine optimal number of clusters using Elbow Method
    """
    wcss = []

    for i in range(1, 10):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Plot elbow graph
    plt.plot(range(1,10), wcss, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.show()


def apply_kmeans(df):
    """
    Apply KMeans clustering
    """
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    return df


def plot_clusters(df):
    """
    Visualize clusters
    """
    plt.scatter(
        df['Annual Income (k$)'],
        df['Spending Score (1-100)'],
        c=df['Cluster'],
        cmap='viridis'
    )
    plt.title("Customer Segmentation")
    plt.xlabel("Income")
    plt.ylabel("Spending")
    plt.show()