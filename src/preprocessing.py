# Preprocessing module for data cleaning and transformation
"""
This module contains functions for data preprocessing.
Handles data loading, cleaning, encoding, normalization, and feature engineering.
"""
# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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
    print("\nBasic Statistics:\n", df.describe())


def clean_data(df):
    """
    Handle missing values and remove duplicates
    """
    # Drop missing values
    df = df.dropna()

    # Remove duplicate rows if any
    df = df.drop_duplicates()

    return df


def encode_gender(df):
    """
    Encode Gender column: Male/M -> 1, Female/F -> 0
    Handles both 'Male'/'Female' and 'M'/'F' formats.
    """
    gender_map = {'Male': 1, 'Female': 0, 'M': 1, 'F': 0}
    df['Gender'] = df['Gender'].map(gender_map)

    # Warn if any values couldn't be mapped
    if df['Gender'].isnull().any():
        unmapped = df['Gender'].isnull().sum()
        print(f"WARNING: {unmapped} Gender values could not be mapped and are now NaN.")

    return df


def normalize_data(df, method='minmax'):
    """
    Normalize numerical columns.

    Parameters:
        df: DataFrame
        method: 'minmax' for Min-Max Scaling, 'standard' for StandardScaler
    
    Returns:
        Normalized DataFrame, fitted scaler object
    """
    cols = ['Annual Income (k$)', 'Spending Score (1-100)']

    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    df[cols] = scaler.fit_transform(df[cols])

    return df, scaler


def add_age_bins(df):
    """
    Bin Age into categorical groups for richer analysis.
    """
    bins = [0, 25, 35, 45, 55, 100]
    labels = ['18-25', '26-35', '36-45', '46-55', '55+']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
    return df


def add_income_category(df, col='Annual Income (k$)'):
    """
    Create income category based on quantiles of the ORIGINAL (unnormalized) data.
    Call this BEFORE normalization.
    """
    df['Income_Category'] = pd.qcut(
        df[col], q=3, labels=['Low', 'Medium', 'High']
    )
    return df


def add_spending_category(df, col='Spending Score (1-100)'):
    """
    Create spending category based on quantiles of the ORIGINAL (unnormalized) data.
    Call this BEFORE normalization.
    """
    df['Spending_Category'] = pd.qcut(
        df[col], q=3, labels=['Low', 'Medium', 'High']
    )
    return df


def save_data(df, path):
    """
    Save dataframe to CSV
    """
    df.to_csv(path, index=False)