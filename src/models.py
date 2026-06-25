# Machine Learning Classification models module
"""
This module contains 7 classification models with comprehensive evaluation.
Includes: Decision Tree, SVM, Random Forest, KNN, Gradient Boosting,
           Gaussian Naive Bayes, Logistic Regression, MLP Neural Network.
Features: k-fold cross-validation, hyperparameter tuning, feature importance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import label_binarize


# ========================================================================
#  DATA PREPARATION (fixes data leakage)
# ========================================================================

def prepare_classification_data(df, target_col='Cluster', test_size=0.2,
                                 random_state=42):
    """
    Prepare data for classification, avoiding data leakage.
    
    Uses ALL features (Age, Gender, Income, Spending) to predict cluster labels,
    where clusters were generated from Income + Spending only. This is a
    legitimate task: predicting segment from ALL available demographics.
    
    Parameters:
        df: DataFrame with features and target column
        target_col: Name of the target column
        test_size: Fraction for test split
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    feature_cols = ['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)']

    # Only use columns that exist in the DataFrame
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_cols


# ========================================================================
#  MODEL DEFINITIONS
# ========================================================================

def get_all_models():
    """
    Return a dict of all classification models with default hyperparameters.
    """
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42
        ),
        'MLP Neural Net': MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42
        ),
    }
    return models


# ========================================================================
#  SINGLE MODEL EVALUATION
# ========================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Train and evaluate a single model with comprehensive metrics.
    
    Returns:
        results: dict with all metrics
        y_pred: predictions on test set
    """
    # Train
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Basic metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # ROC-AUC (handle multi-class)
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            classes = model.classes_
            y_test_bin = label_binarize(y_test, classes=classes)
            if y_test_bin.shape[1] == 1:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr',
                                     average='weighted')
        else:
            auc = None
    except Exception:
        auc = None

    results = {
        'Model': model_name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(auc, 4) if auc is not None else 'N/A'
    }

    return results, y_pred


# ========================================================================
#  K-FOLD CROSS-VALIDATION
# ========================================================================

def cross_validate_model(model, X, y, cv=10, model_name="Model"):
    """
    Perform k-fold cross-validation and return mean ± std scores.
    
    Parameters:
        model: sklearn classifier
        X: Feature matrix
        y: Target vector
        cv: Number of folds
        model_name: Name for display
    
    Returns:
        dict with mean and std of accuracy, f1
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    acc_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')

    return {
        'Model': model_name,
        'CV_Accuracy_Mean': round(acc_scores.mean(), 4),
        'CV_Accuracy_Std': round(acc_scores.std(), 4),
        'CV_F1_Mean': round(f1_scores.mean(), 4),
        'CV_F1_Std': round(f1_scores.std(), 4),
        'cv_acc_scores': acc_scores  # raw scores for statistical tests
    }


# ========================================================================
#  COMPARE ALL MODELS
# ========================================================================

def compare_classifiers(X_train, X_test, y_train, y_test, X_full=None,
                         y_full=None, cv=10):
    """
    Train, evaluate, and cross-validate ALL classification models.
    
    Parameters:
        X_train, X_test, y_train, y_test: Split data
        X_full, y_full: Full dataset for cross-validation (optional)
        cv: Number of CV folds
    
    Returns:
        results_df: Comparison DataFrame (test set metrics)
        cv_df: Cross-validation results DataFrame
        trained_models: Dict of fitted models
    """
    if X_full is None:
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test])

    models = get_all_models()
    test_results = []
    cv_results = []
    trained_models = {}

    for name, model in models.items():
        print(f"  Training {name}...")

        # Test set evaluation
        res, _ = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        test_results.append(res)
        trained_models[name] = model

        # Cross-validation (on fresh copy of model)
        import sklearn.base
        fresh_model = sklearn.base.clone(model)
        cv_res = cross_validate_model(fresh_model, X_full, y_full, cv, name)
        cv_results.append(cv_res)

    # Build DataFrames
    results_df = pd.DataFrame(test_results).sort_values('F1-Score',
                                                         ascending=False)
    cv_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'cv_acc_scores'}
                          for r in cv_results]).sort_values('CV_F1_Mean',
                                                            ascending=False)

    # Display
    print("\n" + "="*80)
    print("  CLASSIFICATION RESULTS (Test Set)")
    print("="*80)
    print(results_df.to_string(index=False))

    print("\n" + "="*80)
    print(f"  CROSS-VALIDATION RESULTS ({cv}-Fold)")
    print("="*80)
    print(cv_df.to_string(index=False))
    print("="*80)

    return results_df, cv_df, trained_models


# ========================================================================
#  FEATURE IMPORTANCE
# ========================================================================

def plot_feature_importance(model, feature_names, model_name="Model",
                             save_path=None):
    """
    Plot feature importance from tree-based models.
    Works with: Decision Tree, Random Forest, Gradient Boosting.
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"  {model_name} does not have feature_importances_. Skipping.")
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']
    bars = plt.bar(range(len(importances)), importances[indices],
                   color=colors[:len(importances)], edgecolor='white',
                   linewidth=0.8)
    plt.xticks(range(len(importances)),
               [feature_names[i] for i in indices], rotation=15, fontsize=10)
    plt.title(f"Feature Importance ({model_name})", fontsize=14, fontweight='bold')
    plt.ylabel("Importance", fontsize=11)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    # Print ranking
    print(f"\n  Feature Importance ({model_name}):")
    for i in indices:
        print(f"    {feature_names[i]}: {importances[i]:.4f}")

    return dict(zip(feature_names, importances))


# ========================================================================
#  HYPERPARAMETER TUNING
# ========================================================================

def tune_random_forest(X_train, y_train, cv=5):
    """
    Tune Random Forest using GridSearchCV.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)

    print(f"\n  Best Random Forest params: {grid.best_params_}")
    print(f"  Best CV F1: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def tune_gradient_boosting(X_train, y_train, cv=5):
    """
    Tune Gradient Boosting using GridSearchCV.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
    }

    grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)

    print(f"\n  Best Gradient Boosting params: {grid.best_params_}")
    print(f"  Best CV F1: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def tune_knn(X_train, y_train, cv=5):
    """
    Tune KNN using GridSearchCV.
    """
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    }

    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)

    print(f"\n  Best KNN params: {grid.best_params_}")
    print(f"  Best CV F1: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_


# ========================================================================
#  VISUALIZATION
# ========================================================================

def plot_model_comparison(results_df, metric='F1-Score', save_path=None):
    """
    Bar chart comparing all models on a given metric.
    """
    df = results_df.copy()

    # Handle 'N/A' values
    if df[metric].dtype == object:
        df = df[df[metric] != 'N/A']
        df[metric] = df[metric].astype(float)

    df = df.sort_values(metric, ascending=True)

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))
    bars = plt.barh(df['Model'], df[metric], color=colors, edgecolor='white',
                    linewidth=0.8, height=0.6)

    # Add value labels
    for bar, val in zip(bars, df[metric]):
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

    plt.title(f"Model Comparison — {metric}", fontsize=14, fontweight='bold')
    plt.xlabel(metric, fontsize=11)
    plt.xlim(0, df[metric].max() * 1.15)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrices(trained_models, X_test, y_test, save_path=None):
    """
    Plot confusion matrices for all models in a grid.
    """
    n_models = len(trained_models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flat if n_models > 1 else [axes]

    for idx, (name, model) in enumerate(trained_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        im = axes[idx].imshow(cm, interpolation='nearest', cmap='Blues')
        axes[idx].set_title(name, fontsize=11, fontweight='bold')

        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[idx].text(j, i, str(cm[i, j]),
                              ha='center', va='center', fontsize=9)

        axes[idx].set_ylabel('True')
        axes[idx].set_xlabel('Predicted')

    # Hide unused subplots
    for idx in range(n_models, len(list(axes))):
        axes[idx].set_visible(False)

    plt.suptitle("Confusion Matrices", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()