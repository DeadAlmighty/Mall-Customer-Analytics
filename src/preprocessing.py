# Preprocessing module for data cleaning and transformation
"""
This module contains functions for data preprocessing.
"""
# Import required libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(path):
    """
    Load dataset from given file path
    """
    df = pd.read_csv(path)
    return df


def explore_data(df):
    """
    Print basic information about dataset
    """
    print("First 5 rows:\n", df.head())
    print("\nDataset Info:\n")
    print(df.info())
    print("\nMissing Values:\n", df.isnull().sum())


def clean_data(df):
    """
    Handle missing values and remove duplicates
    """
    # Drop missing values
    df = df.dropna()

    # Remove duplicate rows if any
    df = df.drop_duplicates()

    return df


def normalize_data(df):
    """
    Normalize numerical columns using Min-Max Scaling
    """
    scaler = MinMaxScaler()

    # Normalize only numeric columns
    df[['Annual Income (k$)', 'Spending Score (1-100)']] = scaler.fit_transform(
        df[['Annual Income (k$)', 'Spending Score (1-100)']]
    )

    return df


def save_data(df, path):
    """
    Save dataframe to CSV
    """
    df.to_csv(path, index=False)