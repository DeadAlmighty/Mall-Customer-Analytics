# Advanced Analysis module
"""
This module contains advanced analysis techniques for paper-worthy results.
Includes: t-SNE, Outlier Detection, Statistical Significance Tests,
           and comprehensive comparison table generators.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats


# ========================================================================
#  t-SNE VISUALIZATION
# ========================================================================

def plot_tsne(X, labels=None, perplexity=30, random_state=42, save_path=None):
    """
    Apply t-SNE dimensionality reduction and plot 2D visualization.
    
    Parameters:
        X: Feature matrix (can be > 2D)
        labels: Optional cluster labels for coloring
        perplexity: t-SNE perplexity parameter
        random_state: Random seed
        save_path: Path to save the plot
    
    Returns:
        tsne_result: 2D t-SNE coordinates (n_samples, 2)
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state,
                max_iter=1000)
    tsne_result = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))

    if labels is not None:
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1],
                              c=labels, cmap='viridis', alpha=0.7,
                              edgecolors='white', linewidth=0.5, s=50)
        plt.colorbar(scatter, label='Cluster')
    else:
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7,
                    color='#2196F3', edgecolors='white', linewidth=0.5, s=50)

    plt.title("t-SNE Visualization", fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=11)
    plt.ylabel("t-SNE Dimension 2", fontsize=11)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return tsne_result


def plot_pca_vs_tsne(X, labels=None, save_path=None):
    """
    Side-by-side comparison of PCA and t-SNE dimensionality reduction.
    """
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    tsne_result = tsne.fit_transform(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    if labels is not None:
        ax1.scatter(pca_result[:, 0], pca_result[:, 1], c=labels,
                    cmap='viridis', alpha=0.7, edgecolors='white',
                    linewidth=0.5, s=50)
        ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels,
                    cmap='viridis', alpha=0.7, edgecolors='white',
                    linewidth=0.5, s=50)
    else:
        ax1.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7,
                    color='#2196F3', edgecolors='white', linewidth=0.5, s=50)
        ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7,
                    color='#E91E63', edgecolors='white', linewidth=0.5, s=50)

    ax1.set_title("PCA", fontsize=14, fontweight='bold')
    ax1.set_xlabel("PC1", fontsize=11)
    ax1.set_ylabel("PC2", fontsize=11)
    ax1.grid(True, alpha=0.2)

    var_explained = pca.explained_variance_ratio_
    ax1.text(0.02, 0.98, f"Var explained: {sum(var_explained)*100:.1f}%",
             transform=ax1.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.set_title("t-SNE", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Dim 1", fontsize=11)
    ax2.set_ylabel("Dim 2", fontsize=11)
    ax2.grid(True, alpha=0.2)

    plt.suptitle("PCA vs t-SNE Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return pca_result, tsne_result


# ========================================================================
#  OUTLIER DETECTION
# ========================================================================

def detect_outliers_isolation_forest(X, contamination=0.05):
    """
    Detect outliers using Isolation Forest.
    
    Parameters:
        X: Feature matrix
        contamination: Expected proportion of outliers
    
    Returns:
        outlier_mask: Boolean array (True = outlier)
        scores: Anomaly scores
    """
    iso = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso.fit_predict(X)  # 1 = inlier, -1 = outlier
    scores = iso.decision_function(X)

    outlier_mask = predictions == -1
    n_outliers = outlier_mask.sum()

    print(f"\n  Isolation Forest: {n_outliers} outliers detected "
          f"({n_outliers/len(X)*100:.1f}%)")

    return outlier_mask, scores


def detect_outliers_lof(X, n_neighbors=20, contamination=0.05):
    """
    Detect outliers using Local Outlier Factor (LOF).
    
    Parameters:
        X: Feature matrix
        n_neighbors: Number of neighbors for LOF
        contamination: Expected proportion of outliers
    
    Returns:
        outlier_mask: Boolean array (True = outlier)
        scores: Negative outlier factor scores
    """
    lof = LocalOutlierFactor(n_neighbors=n_neighbors,
                              contamination=contamination)
    predictions = lof.fit_predict(X)
    scores = lof.negative_outlier_factor_

    outlier_mask = predictions == -1
    n_outliers = outlier_mask.sum()

    print(f"  LOF: {n_outliers} outliers detected "
          f"({n_outliers/len(X)*100:.1f}%)")

    return outlier_mask, scores


def plot_outliers(X, outlier_mask_iso, outlier_mask_lof,
                   xlabel="Income", ylabel="Spending", save_path=None):
    """
    Visualize outliers detected by both methods side by side.
    Expects X to have at least 2 columns.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if isinstance(X, pd.DataFrame):
        x1 = X.iloc[:, 0]
        x2 = X.iloc[:, 1]
    else:
        x1 = X[:, 0]
        x2 = X[:, 1]

    # Isolation Forest
    ax1.scatter(x1[~outlier_mask_iso], x2[~outlier_mask_iso],
                alpha=0.5, s=30, c='#2196F3', label='Inlier')
    ax1.scatter(x1[outlier_mask_iso], x2[outlier_mask_iso],
                alpha=0.9, s=80, c='red', marker='X', label='Outlier')
    ax1.set_title("Isolation Forest", fontsize=13, fontweight='bold')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # LOF
    ax2.scatter(x1[~outlier_mask_lof], x2[~outlier_mask_lof],
                alpha=0.5, s=30, c='#4CAF50', label='Inlier')
    ax2.scatter(x1[outlier_mask_lof], x2[outlier_mask_lof],
                alpha=0.9, s=80, c='red', marker='X', label='Outlier')
    ax2.set_title("Local Outlier Factor (LOF)", fontsize=13, fontweight='bold')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    plt.suptitle("Outlier Detection Comparison", fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ========================================================================
#  STATISTICAL SIGNIFICANCE TESTING
# ========================================================================

def paired_ttest(cv_scores_a, cv_scores_b, name_a="Model A", name_b="Model B"):
    """
    Perform a paired t-test between two models' cross-validation scores.
    
    Parameters:
        cv_scores_a, cv_scores_b: Arrays of CV scores from the same folds
        name_a, name_b: Model names for display
    
    Returns:
        t_stat, p_value, is_significant (at α=0.05)
    """
    t_stat, p_value = stats.ttest_rel(cv_scores_a, cv_scores_b)
    is_significant = p_value < 0.05

    print(f"\n  Paired t-test: {name_a} vs {name_b}")
    print(f"    t-statistic: {t_stat:.4f}")
    print(f"    p-value:     {p_value:.4f}")
    print(f"    Significant: {'YES ✓' if is_significant else 'NO ✗'} (α=0.05)")

    return t_stat, p_value, is_significant


def compare_all_pairs(cv_results):
    """
    Perform paired t-tests between all pairs of models.
    
    Parameters:
        cv_results: List of dicts from cross_validate_model() containing
                    'Model' and 'cv_acc_scores' keys.
    
    Returns:
        DataFrame with p-values for all model pairs
    """
    names = [r['Model'] for r in cv_results]
    n = len(names)

    p_matrix = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            _, p_val = stats.ttest_rel(
                cv_results[i]['cv_acc_scores'],
                cv_results[j]['cv_acc_scores']
            )
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val

    p_df = pd.DataFrame(p_matrix, index=names, columns=names)

    print("\n" + "="*65)
    print("  PAIRWISE p-VALUES (paired t-test)")
    print("  Values < 0.05 indicate statistically significant difference")
    print("="*65)
    print(p_df.round(4).to_string())
    print("="*65)

    return p_df


# ========================================================================
#  COMPREHENSIVE SUMMARY GENERATOR
# ========================================================================

def generate_summary_report(clf_results=None, reg_results=None,
                             cluster_results=None, save_path=None):
    """
    Generate a comprehensive text summary of all results.
    Useful for the paper's Results section.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("  COMPREHENSIVE RESULTS SUMMARY")
    lines.append("=" * 70)

    if clf_results is not None:
        lines.append("\n--- CLASSIFICATION ---")
        best = clf_results.iloc[0]
        lines.append(f"  Best model: {best['Model']} "
                     f"(F1={best['F1-Score']}, Acc={best['Accuracy']})")
        lines.append(f"  Total models evaluated: {len(clf_results)}")
        lines.append(f"  Accuracy range: "
                     f"{clf_results['Accuracy'].min():.3f} - "
                     f"{clf_results['Accuracy'].max():.3f}")

    if reg_results is not None:
        lines.append("\n--- REGRESSION ---")
        best = reg_results.iloc[0]
        lines.append(f"  Best model: {best['Model']} (R²={best['R²']})")
        lines.append(f"  Total models evaluated: {len(reg_results)}")
        lines.append(f"  R² range: {reg_results['R²'].min():.3f} - "
                     f"{reg_results['R²'].max():.3f}")

    if cluster_results is not None:
        lines.append("\n--- CLUSTERING ---")
        valid = cluster_results.dropna(subset=['Silhouette'])
        if len(valid) > 0:
            best = valid.sort_values('Silhouette', ascending=False).iloc[0]
            lines.append(f"  Best method: {best['Model']} "
                         f"(Silhouette={best['Silhouette']})")
        lines.append(f"  Total methods evaluated: {len(cluster_results)}")

    lines.append("\n" + "=" * 70)

    report = "\n".join(lines)
    print(report)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"\n  Report saved to: {save_path}")

    return report
