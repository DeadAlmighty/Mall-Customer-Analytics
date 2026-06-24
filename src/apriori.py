# Association Rules Mining module
"""
This module contains association rules mining using the Apriori algorithm.
Uses median-based and quantile-based categorical splits for meaningful rules.
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def prepare_data(df):
    """
    Convert dataset into boolean categorical format for Apriori.
    Uses MEDIAN-BASED splits to create meaningful binary categories.
    
    Parameters:
        df: DataFrame with original (unnormalized) or normalized columns.
            Works correctly regardless because median adapts to the data.
    
    Returns:
        DataFrame with boolean columns suitable for Apriori.
    """
    basket = pd.DataFrame()

    # --- Income categories (median split) ---
    income_median = df['Annual Income (k$)'].median()
    basket['High_Income'] = df['Annual Income (k$)'] > income_median
    basket['Low_Income'] = df['Annual Income (k$)'] <= income_median

    # --- Spending categories (median split) ---
    spending_median = df['Spending Score (1-100)'].median()
    basket['High_Spending'] = df['Spending Score (1-100)'] > spending_median
    basket['Low_Spending'] = df['Spending Score (1-100)'] <= spending_median

    # --- Age categories (quantile-based) ---
    age_median = df['Age'].median()
    basket['Young'] = df['Age'] <= age_median
    basket['Senior'] = df['Age'] > age_median

    # --- Gender ---
    if 'Gender' in df.columns:
        # Handle both encoded (0/1) and string ('M'/'F'/'Male'/'Female') formats
        if df['Gender'].dtype == 'object':
            basket['Male'] = df['Gender'].isin(['M', 'Male'])
            basket['Female'] = df['Gender'].isin(['F', 'Female'])
        else:
            basket['Male'] = df['Gender'] == 1
            basket['Female'] = df['Gender'] == 0

    return basket


def prepare_data_multibin(df):
    """
    Convert dataset into multi-level boolean categories for richer rules.
    Uses TERCILE (3-bin) splits.
    
    Returns:
        DataFrame with boolean columns suitable for Apriori.
    """
    basket = pd.DataFrame()

    # --- Income terciles ---
    income_q = df['Annual Income (k$)'].quantile([0.33, 0.66])
    basket['Income_Low'] = df['Annual Income (k$)'] <= income_q[0.33]
    basket['Income_Mid'] = (df['Annual Income (k$)'] > income_q[0.33]) & \
                           (df['Annual Income (k$)'] <= income_q[0.66])
    basket['Income_High'] = df['Annual Income (k$)'] > income_q[0.66]

    # --- Spending terciles ---
    spend_q = df['Spending Score (1-100)'].quantile([0.33, 0.66])
    basket['Spending_Low'] = df['Spending Score (1-100)'] <= spend_q[0.33]
    basket['Spending_Mid'] = (df['Spending Score (1-100)'] > spend_q[0.33]) & \
                             (df['Spending Score (1-100)'] <= spend_q[0.66])
    basket['Spending_High'] = df['Spending Score (1-100)'] > spend_q[0.66]

    # --- Age groups ---
    basket['Age_Young'] = df['Age'] <= 30
    basket['Age_Mid'] = (df['Age'] > 30) & (df['Age'] <= 45)
    basket['Age_Senior'] = df['Age'] > 45

    # --- Gender ---
    if 'Gender' in df.columns:
        if df['Gender'].dtype == 'object':
            basket['Male'] = df['Gender'].isin(['M', 'Male'])
            basket['Female'] = df['Gender'].isin(['F', 'Female'])
        else:
            basket['Male'] = df['Gender'] == 1
            basket['Female'] = df['Gender'] == 0

    return basket


def apply_apriori(df, min_support=0.15, min_confidence=0.5):
    """
    Apply Apriori algorithm to find frequent itemsets and association rules.
    
    Parameters:
        df: Boolean DataFrame (output of prepare_data or prepare_data_multibin)
        min_support: Minimum support threshold
        min_confidence: Minimum confidence threshold
    
    Returns:
        frequent_items: DataFrame of frequent itemsets
        rules: DataFrame of association rules sorted by lift
    """
    frequent_items = apriori(df, min_support=min_support, use_colnames=True)

    if len(frequent_items) == 0:
        print("No frequent itemsets found. Try lowering min_support.")
        return frequent_items, pd.DataFrame()

    rules = association_rules(
        frequent_items, metric="confidence", min_threshold=min_confidence
    )

    # Sort by lift (most interesting rules first)
    if len(rules) > 0:
        rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)

    return frequent_items, rules


def print_top_rules(rules, n=10):
    """
    Pretty-print the top N association rules.
    """
    if len(rules) == 0:
        print("No rules to display.")
        return

    print(f"\n{'='*70}")
    print(f"  TOP {min(n, len(rules))} ASSOCIATION RULES (sorted by Lift)")
    print(f"{'='*70}")

    for i, row in rules.head(n).iterrows():
        ant = ', '.join(list(row['antecedents']))
        con = ', '.join(list(row['consequents']))
        print(f"\n  Rule {i+1}: {ant}  -->  {con}")
        print(f"    Support:    {row['support']:.3f}")
        print(f"    Confidence: {row['confidence']:.3f}")
        print(f"    Lift:       {row['lift']:.3f}")

    print(f"\n{'='*70}")