# Exploratory Data Analysis module
"""
This module contains functions for exploratory data analysis.
Includes distribution plots, correlation analysis, and demographic breakdowns.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_age_distribution(df, save_path=None):
    """
    Plot age distribution
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Age'], bins=20, kde=True, color='#2196F3',
                 edgecolor='white')
    plt.title("Age Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Age", fontsize=11)
    plt.ylabel("Count", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_correlation(df, save_path=None):
    """
    Plot correlation heatmap for numerical columns
    """
    numeric_df = df.select_dtypes(include='number')
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0,
                fmt='.2f', linewidths=0.5, square=True)
    plt.title("Correlation Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_income_vs_spending(df, save_path=None):
    """
    Scatter plot of income vs spending with optional gender coloring
    """
    plt.figure(figsize=(8, 6))

    if 'Gender' in df.columns and df['Gender'].dtype == 'object':
        sns.scatterplot(
            x='Annual Income (k$)', y='Spending Score (1-100)',
            hue='Gender', data=df, palette=['#E91E63', '#2196F3'],
            alpha=0.7, edgecolor='white', s=60
        )
    else:
        sns.scatterplot(
            x='Annual Income (k$)', y='Spending Score (1-100)',
            data=df, color='#2196F3', alpha=0.7, edgecolor='white', s=60
        )

    plt.title("Income vs Spending", fontsize=14, fontweight='bold')
    plt.xlabel("Annual Income (k$)", fontsize=11)
    plt.ylabel("Spending Score (1-100)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_gender_distribution(df, save_path=None):
    """
    Plot gender distribution as a pie chart
    """
    plt.figure(figsize=(6, 6))
    gender_col = df['Gender'] if df['Gender'].dtype == 'object' else \
        df['Gender'].map({1: 'Male', 0: 'Female'})
    counts = gender_col.value_counts()
    colors = ['#2196F3', '#E91E63']
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=colors,
            startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
    plt.title("Gender Distribution", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_distributions(df, save_path=None):
    """
    Plot distributions of all numerical columns in a grid
    """
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    # Remove CustomerID if present
    numeric_cols = [c for c in numeric_cols if 'ID' not in c.upper()]

    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flat

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    for idx, col in enumerate(numeric_cols):
        sns.histplot(df[col], bins=20, kde=True, ax=axes[idx],
                     color=colors[idx % len(colors)], edgecolor='white')
        axes[idx].set_title(col, fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)

    for idx in range(len(numeric_cols), len(list(axes))):
        axes[idx].set_visible(False)

    plt.suptitle("Feature Distributions", fontsize=15, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_boxplots(df, save_path=None):
    """
    Box plots for numerical columns to visualize spread and outliers
    """
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    numeric_cols = [c for c in numeric_cols if 'ID' not in c.upper()]

    fig, axes = plt.subplots(1, len(numeric_cols),
                             figsize=(4*len(numeric_cols), 5))
    if len(numeric_cols) == 1:
        axes = [axes]

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    for idx, col in enumerate(numeric_cols):
        bp = axes[idx].boxplot(df[col].dropna(), patch_artist=True)
        bp['boxes'][0].set_facecolor(colors[idx % len(colors)])
        bp['boxes'][0].set_alpha(0.7)
        axes[idx].set_title(col, fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle("Box Plots", fontsize=15, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
