# Association Rules Mining module
"""
This module contains association rules mining algorithms.
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def prepare_data(df):
    """
    Convert dataset into categorical format for Apriori
    """
    df['High_Income'] = df['Annual Income (k$)'] > 0.5
    df['High_Spending'] = df['Spending Score (1-100)'] > 0.5

    return df[['High_Income', 'High_Spending']]


def apply_apriori(df):
    """
    Apply Apriori algorithm
    """
    frequent_items = apriori(df, min_support=0.2, use_colnames=True)
    rules = association_rules(frequent_items, metric="confidence", min_threshold=0.5)

    return rules