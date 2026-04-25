# Exploratory Data Analysis module
"""
This module contains functions for exploratory data analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns

def plot_age_distribution(df, save_path):
    """
    Plot age distribution
    """
    plt.figure(figsize=(6,4))
    sns.histplot(df['Age'], bins=20, kde=True)
    plt.title("Age Distribution")
    plt.savefig(save_path)
    plt.show()


def plot_correlation(df, save_path):
    """
    Plot correlation heatmap
    """
    plt.figure(figsize=(6,4))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.savefig(save_path)
    plt.show()


def plot_income_vs_spending(df, save_path):
    """
    Scatter plot of income vs spending
    """
    plt.figure(figsize=(6,4))
    sns.scatterplot(
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        data=df
    )
    plt.title("Income vs Spending")
    plt.savefig(save_path)
    plt.show()
