# Regression module
"""
This module contains multiple regression models with proper evaluation.
Includes: Linear, Polynomial, Ridge, Lasso, Random Forest, SVR.
All models use train/test split and report R², MSE, RMSE, MAE.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)


# ========================================================================
#  DATA PREPARATION
# ========================================================================

def prepare_regression_data(df, feature_cols=None, target_col='Spending Score (1-100)',
                             test_size=0.2, random_state=42):
    """
    Prepare data for regression with proper train/test split.
    
    Parameters:
        df: DataFrame
        feature_cols: List of feature column names. Defaults to ['Annual Income (k$)']
        target_col: Target column name
        test_size: Fraction for test split
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    if feature_cols is None:
        feature_cols = ['Annual Income (k$)']

    # Only use columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, feature_cols


# ========================================================================
#  MODEL DEFINITIONS
# ========================================================================

def get_all_regressors(poly_degree=2):
    """
    Return a dict of all regression models.
    """
    models = {
        'Linear Regression': LinearRegression(),
        f'Polynomial (deg={poly_degree})': make_pipeline(
            PolynomialFeatures(degree=poly_degree, include_bias=False),
            LinearRegression()
        ),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1, max_iter=5000),
        'Decision Tree Regressor': DecisionTreeRegressor(
            max_depth=5, random_state=42
        ),
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=42
        ),
        'SVR': SVR(kernel='rbf', C=100, gamma='scale'),
    }
    return models


# ========================================================================
#  SINGLE MODEL EVALUATION
# ========================================================================

def evaluate_regressor(model, X_train, X_test, y_train, y_test,
                        model_name="Model"):
    """
    Train and evaluate a single regression model.
    Reports R², MSE, RMSE, MAE, and Adjusted R².
    
    Returns:
        results: dict with all metrics
        model: fitted model
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    # Adjusted R²
    n = len(y_test)
    p = X_test.shape[1]
    if n - p - 1 > 0:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        adj_r2 = r2

    results = {
        'Model': model_name,
        'R²': round(r2, 4),
        'Adjusted R²': round(adj_r2, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
    }

    return results, model


# ========================================================================
#  COMPARE ALL MODELS
# ========================================================================

def compare_regressors(X_train, X_test, y_train, y_test, poly_degree=2,
                        cv=5):
    """
    Train, evaluate, and cross-validate ALL regression models.
    
    Returns:
        results_df: Comparison DataFrame
        trained_models: Dict of fitted models
    """
    models = get_all_regressors(poly_degree)
    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"  Training {name}...")
        res, fitted = evaluate_regressor(
            model, X_train, X_test, y_train, y_test, name
        )

        # Cross-validation R²
        import sklearn.base
        fresh = sklearn.base.clone(model)
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test])
        try:
            cv_scores = cross_val_score(fresh, X_full, y_full, cv=cv,
                                         scoring='r2')
            res['CV_R²_Mean'] = round(cv_scores.mean(), 4)
            res['CV_R²_Std'] = round(cv_scores.std(), 4)
        except Exception:
            res['CV_R²_Mean'] = 'N/A'
            res['CV_R²_Std'] = 'N/A'

        results.append(res)
        trained_models[name] = fitted

    results_df = pd.DataFrame(results).sort_values('R²', ascending=False)

    print("\n" + "="*90)
    print("  REGRESSION RESULTS")
    print("="*90)
    print(results_df.to_string(index=False))
    print("="*90)

    return results_df, trained_models


# ========================================================================
#  VISUALIZATION
# ========================================================================

def plot_regression(model, X_train, X_test, y_train, y_test,
                     model_name="Model", save_path=None):
    """
    Plot actual vs predicted values with regression line.
    For 1D features, shows scatter + line. For multi-D, shows actual vs predicted.
    """
    y_pred_test = model.predict(X_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Actual vs Predicted ---
    ax1.scatter(y_test, y_pred_test, alpha=0.6, color='#2196F3',
                edgecolors='white', linewidth=0.5, s=50)

    # Perfect prediction line
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
             label='Perfect Prediction')

    ax1.set_title(f"Actual vs Predicted ({model_name})", fontsize=13,
                  fontweight='bold')
    ax1.set_xlabel("Actual", fontsize=11)
    ax1.set_ylabel("Predicted", fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Residuals ---
    residuals = y_test - y_pred_test
    ax2.scatter(y_pred_test, residuals, alpha=0.6, color='#FF5722',
                edgecolors='white', linewidth=0.5, s=50)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_title(f"Residual Plot ({model_name})", fontsize=13,
                  fontweight='bold')
    ax2.set_xlabel("Predicted", fontsize=11)
    ax2.set_ylabel("Residual", fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_regression_comparison(results_df, save_path=None):
    """
    Bar chart comparing all regression models on R².
    """
    df = results_df.copy()
    df = df.sort_values('R²', ascending=True)

    plt.figure(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df)))
    bars = plt.barh(df['Model'], df['R²'], color=colors, edgecolor='white',
                    linewidth=0.8, height=0.6)

    for bar, val in zip(bars, df['R²']):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

    plt.title("Regression Model Comparison — R²", fontsize=14, fontweight='bold')
    plt.xlabel("R² Score", fontsize=11)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_all_regressions_1d(trained_models, X_train, X_test, y_train, y_test,
                             save_path=None):
    """
    Plot all regression models' predictions on the same scatter plot.
    Only works for 1D feature input.
    """
    if X_test.shape[1] != 1:
        print("  Skipping 1D plot: multiple features detected.")
        return

    plt.figure(figsize=(12, 7))
    plt.scatter(X_test.values.ravel(), y_test, alpha=0.4, color='gray',
                label='Actual', s=40, edgecolors='white')

    colors = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800', '#9C27B0',
              '#00BCD4', '#795548']

    # Sort X for smooth line plots
    sort_idx = np.argsort(X_test.values.ravel())
    X_sorted = X_test.iloc[sort_idx]

    for idx, (name, model) in enumerate(trained_models.items()):
        y_pred_sorted = model.predict(X_sorted)
        plt.plot(X_sorted.values.ravel(), y_pred_sorted,
                 color=colors[idx % len(colors)], linewidth=2,
                 label=name, alpha=0.8)

    plt.title("All Regression Models", fontsize=14, fontweight='bold')
    plt.xlabel("Annual Income (k$)", fontsize=11)
    plt.ylabel("Spending Score", fontsize=11)
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
