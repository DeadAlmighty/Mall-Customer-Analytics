# Regression module
"""
This module contains regression models.
"""

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def train_regression(df):
    """
    Train Linear Regression model
    """
    X = df[['Annual Income (k$)']]
    y = df['Spending Score (1-100)']

    model = LinearRegression()
    model.fit(X, y)

    return model, X, y


def plot_regression(model, X, y):
    """
    Plot regression line
    """
    plt.scatter(X, y)
    plt.plot(X, model.predict(X), color='red')
    plt.title("Regression Analysis")
    plt.xlabel("Income")
    plt.ylabel("Spending")
    plt.show()
