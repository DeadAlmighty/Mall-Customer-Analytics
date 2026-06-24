# Clustering module
"""
This module contains multiple clustering algorithms and evaluation metrics.
Includes: KMeans, DBSCAN, Agglomerative (Hierarchical), Gaussian Mixture Model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage


# ========================================================================
#  KMEANS
# ========================================================================

def elbow_method(X, k_range=range(2, 11)):
    """
    Determine optimal number of clusters using Elbow Method.
    Also computes Silhouette Scores for each k.
    """
    wcss = []
    sil_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        wcss.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X, labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow plot
    ax1.plot(list(k_range), wcss, marker='o', color='#2196F3', linewidth=2)
    ax1.set_title("Elbow Method", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("WCSS (Inertia)")
    ax1.grid(True, alpha=0.3)

    # Silhouette plot
    ax2.plot(list(k_range), sil_scores, marker='s', color='#4CAF50', linewidth=2)
    ax2.set_title("Silhouette Score vs k", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/plots/elbow_silhouette.png', dpi=150, bbox_inches='tight')
    plt.show()

    best_k = list(k_range)[np.argmax(sil_scores)]
    print(f"\nBest k by Silhouette Score: {best_k} (score={max(sil_scores):.4f})")

    return wcss, sil_scores


def apply_kmeans(X, n_clusters=5):
    """
    Apply KMeans clustering.
    
    Parameters:
        X: Feature matrix (numpy array or DataFrame)
        n_clusters: Number of clusters
    
    Returns:
        labels: Cluster assignments
        model: Fitted KMeans model
    """
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    return labels, model


# ========================================================================
#  DBSCAN
# ========================================================================

def apply_dbscan(X, eps=0.3, min_samples=5):
    """
    Apply DBSCAN clustering.
    Automatically detects number of clusters and identifies noise points.
    
    Parameters:
        X: Feature matrix (should be normalized/scaled)
        eps: Maximum distance between two samples
        min_samples: Minimum points to form a dense region
    
    Returns:
        labels: Cluster assignments (-1 = noise)
        model: Fitted DBSCAN model
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"DBSCAN: {n_clusters} clusters found, {n_noise} noise points")

    return labels, model


# ========================================================================
#  HIERARCHICAL (AGGLOMERATIVE)
# ========================================================================

def apply_hierarchical(X, n_clusters=5, linkage_type='ward'):
    """
    Apply Agglomerative (Hierarchical) clustering.
    
    Parameters:
        X: Feature matrix
        n_clusters: Number of clusters
        linkage_type: 'ward', 'complete', 'average', 'single'
    
    Returns:
        labels: Cluster assignments
        model: Fitted AgglomerativeClustering model
    """
    model = AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkage_type
    )
    labels = model.fit_predict(X)
    return labels, model


def plot_dendrogram(X, method='ward', max_display=30):
    """
    Plot a dendrogram for hierarchical clustering.
    """
    Z = linkage(X, method=method)

    plt.figure(figsize=(14, 6))
    dendrogram(Z, truncate_mode='lastp', p=max_display,
               leaf_rotation=90, leaf_font_size=8,
               show_contracted=True, color_threshold=0)
    plt.title("Hierarchical Clustering Dendrogram", fontsize=14, fontweight='bold')
    plt.xlabel("Sample Index (or Cluster Size)")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig('outputs/plots/dendrogram.png', dpi=150, bbox_inches='tight')
    plt.show()


# ========================================================================
#  GAUSSIAN MIXTURE MODEL
# ========================================================================

def apply_gmm(X, n_components=5):
    """
    Apply Gaussian Mixture Model clustering.
    Provides soft (probabilistic) cluster assignments.
    
    Parameters:
        X: Feature matrix
        n_components: Number of mixture components (clusters)
    
    Returns:
        labels: Hard cluster assignments
        model: Fitted GaussianMixture model
        probs: Soft assignment probabilities (n_samples x n_components)
    """
    model = GaussianMixture(
        n_components=n_components, random_state=42, covariance_type='full'
    )
    model.fit(X)
    labels = model.predict(X)
    probs = model.predict_proba(X)

    print(f"GMM: BIC = {model.bic(X):.2f}, AIC = {model.aic(X):.2f}")

    return labels, model, probs


# ========================================================================
#  EVALUATION & COMPARISON
# ========================================================================

def evaluate_clustering(X, labels, name="Model"):
    """
    Evaluate clustering quality using multiple metrics.
    Returns a dict of metrics. Handles cases with noise points (DBSCAN).
    """
    # Filter out noise points for metric calculation
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        print(f"  {name}: Not enough clusters for evaluation.")
        return {'Model': name, 'Silhouette': None, 'Davies-Bouldin': None,
                'Calinski-Harabasz': None, 'n_clusters': len(set(labels[mask]))}

    X_clean = X[mask] if isinstance(X, np.ndarray) else X.iloc[mask]
    labels_clean = labels[mask]

    sil = silhouette_score(X_clean, labels_clean)
    dbi = davies_bouldin_score(X_clean, labels_clean)
    chi = calinski_harabasz_score(X_clean, labels_clean)
    n_clust = len(set(labels_clean))

    return {
        'Model': name,
        'n_clusters': n_clust,
        'Silhouette': round(sil, 4),
        'Davies-Bouldin': round(dbi, 4),
        'Calinski-Harabasz': round(chi, 2)
    }


def compare_clustering(X, n_clusters=5, dbscan_eps=0.3, dbscan_min_samples=5):
    """
    Run all clustering algorithms and return a comparison DataFrame.
    
    Parameters:
        X: Feature matrix (should be scaled/normalized)
        n_clusters: Number of clusters for KMeans, Hierarchical, GMM
        dbscan_eps: DBSCAN eps parameter
        dbscan_min_samples: DBSCAN min_samples parameter
    
    Returns:
        comparison_df: DataFrame comparing all methods
        all_labels: Dict mapping method name to labels
    """
    results = []
    all_labels = {}

    # KMeans
    labels_km, _ = apply_kmeans(X, n_clusters)
    all_labels['KMeans'] = labels_km
    results.append(evaluate_clustering(X, labels_km, 'KMeans'))

    # DBSCAN
    labels_db, _ = apply_dbscan(X, dbscan_eps, dbscan_min_samples)
    all_labels['DBSCAN'] = labels_db
    results.append(evaluate_clustering(X, labels_db, 'DBSCAN'))

    # Hierarchical
    labels_hc, _ = apply_hierarchical(X, n_clusters)
    all_labels['Hierarchical'] = labels_hc
    results.append(evaluate_clustering(X, labels_hc, 'Hierarchical'))

    # GMM
    labels_gmm, _, _ = apply_gmm(X, n_clusters)
    all_labels['GMM'] = labels_gmm
    results.append(evaluate_clustering(X, labels_gmm, 'GMM'))

    comparison_df = pd.DataFrame(results)
    print("\n" + "="*65)
    print("  CLUSTERING COMPARISON")
    print("="*65)
    print(comparison_df.to_string(index=False))
    print("="*65)

    return comparison_df, all_labels


# ========================================================================
#  VISUALIZATION
# ========================================================================

def plot_clusters(X, labels, title="Clusters", xlabel="Feature 1",
                  ylabel="Feature 2", save_path=None):
    """
    Visualize 2D clustering results.
    """
    plt.figure(figsize=(8, 6))

    unique_labels = sorted(set(labels))
    cmap = plt.colormaps.get_cmap('viridis').resampled(len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        if isinstance(X, pd.DataFrame):
            x_vals = X.iloc[mask, 0]
            y_vals = X.iloc[mask, 1]
        else:
            x_vals = X[mask, 0]
            y_vals = X[mask, 1]

        if label == -1:
            plt.scatter(x_vals, y_vals, c='gray', marker='x',
                        s=30, alpha=0.5, label='Noise')
        else:
            plt.scatter(x_vals, y_vals, c=[cmap(i)], s=50, alpha=0.7,
                        label=f'Cluster {label}', edgecolors='white', linewidth=0.5)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_all_clusters(X, all_labels, xlabel="Income", ylabel="Spending"):
    """
    Plot all clustering results side by side in a 2x2 grid.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    methods = list(all_labels.keys())

    for idx, (ax, method) in enumerate(zip(axes.flat, methods)):
        labels = all_labels[method]
        unique = sorted(set(labels))
        cmap = plt.colormaps.get_cmap('viridis').resampled(len(unique))

        for i, label in enumerate(unique):
            mask = labels == label
            if isinstance(X, pd.DataFrame):
                x_vals = X.iloc[mask, 0]
                y_vals = X.iloc[mask, 1]
            else:
                x_vals = X[mask, 0]
                y_vals = X[mask, 1]

            if label == -1:
                ax.scatter(x_vals, y_vals, c='gray', marker='x', s=20, alpha=0.5)
            else:
                ax.scatter(x_vals, y_vals, c=[cmap(i)], s=30, alpha=0.7,
                           edgecolors='white', linewidth=0.3)

        ax.set_title(method, fontsize=13, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.2)

    plt.suptitle("Clustering Methods Comparison", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('outputs/plots/clustering_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()