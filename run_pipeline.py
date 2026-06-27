"""Temporary runner - executes the full pipeline"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.preprocessing import (
    load_data, clean_data, encode_gender, normalize_data,
    add_age_bins, add_income_category, add_spending_category
)
from src.eda import (
    plot_age_distribution, plot_correlation, plot_income_vs_spending,
    plot_gender_distribution, plot_distributions, plot_boxplots
)
from src.clustering import (
    elbow_method, compare_clustering, plot_all_clusters,
    plot_dendrogram, apply_kmeans
)
from src.models import (
    prepare_classification_data, compare_classifiers,
    plot_feature_importance, plot_model_comparison,
    plot_confusion_matrices,
    tune_random_forest, tune_gradient_boosting, tune_knn,
    cross_validate_model, get_all_models
)
from src.regression import (
    prepare_regression_data, compare_regressors,
    plot_regression_comparison, plot_all_regressions_1d, plot_regression
)
from src.apriori import (
    prepare_data, prepare_data_multibin, apply_apriori, print_top_rules
)
from src.advanced_analysis import (
    plot_pca_vs_tsne, detect_outliers_isolation_forest,
    detect_outliers_lof, plot_outliers, compare_all_pairs,
    generate_summary_report
)

os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)

print("="*70)
print("  DMW PROJECT - FULL PIPELINE")
print("="*70)

# === STEP 1: PREPROCESSING ===
print("\n[STEP 1] Data Loading & Preprocessing")
df = load_data('data/raw/mall_customers.csv')
print(f"  Raw shape: {df.shape}")
df = clean_data(df)
print(f"  After cleaning: {df.shape}")
df_original = df.copy()
df_original = add_age_bins(df_original)
df_original = add_income_category(df_original)
df_original = add_spending_category(df_original)
df = encode_gender(df)
df_normalized, scaler = normalize_data(df, method='minmax')
print("  [OK] Preprocessing complete")

# === STEP 2: EDA ===
print("\n[STEP 2] Exploratory Data Analysis")
plot_distributions(df_original, save_path='outputs/plots/feature_distributions.png')
plot_age_distribution(df_original, save_path='outputs/plots/age_distribution.png')
plot_income_vs_spending(df_original, save_path='outputs/plots/income_vs_spending.png')
plot_gender_distribution(df_original, save_path='outputs/plots/gender_distribution.png')
plot_correlation(df_normalized, save_path='outputs/plots/correlation_matrix.png')
plot_boxplots(df_original, save_path='outputs/plots/boxplots.png')
print("  [OK] 6 EDA plots saved")

# === STEP 3: CLUSTERING ===
print("\n[STEP 3] Clustering Analysis")
X_cluster_2d = df_normalized[['Annual Income (k$)', 'Spending Score (1-100)']].values
wcss, sil_scores = elbow_method(X_cluster_2d)
plot_dendrogram(X_cluster_2d)
cluster_results_df, all_labels = compare_clustering(
    X_cluster_2d, n_clusters=5, dbscan_eps=0.15, dbscan_min_samples=5
)
cluster_results_df.to_csv('outputs/results/clustering_comparison.csv', index=False)
plot_all_clusters(X_cluster_2d, all_labels, xlabel="Income", ylabel="Spending")

X_cluster_4d = df_normalized[['Age', 'Gender', 'Annual Income (k$)',
                               'Spending Score (1-100)']].values
cluster_results_4d, all_labels_4d = compare_clustering(
    X_cluster_4d, n_clusters=5, dbscan_eps=0.4, dbscan_min_samples=5
)

df_normalized['Cluster'] = all_labels['KMeans']
df_original['Cluster'] = all_labels['KMeans']
print("  [OK] Clustering complete")

# === STEP 4: CLASSIFICATION ===
print("\n[STEP 4] Classification (8 models)")
X_train, X_test, y_train, y_test, feature_names = prepare_classification_data(
    df_normalized, target_col='Cluster'
)
X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])

clf_results_df, cv_results_df, trained_models = compare_classifiers(
    X_train, X_test, y_train, y_test, X_full, y_full, cv=10
)
clf_results_df.to_csv('outputs/results/classification_comparison.csv', index=False)
cv_results_df.to_csv('outputs/results/classification_cv_results.csv', index=False)

plot_model_comparison(clf_results_df, metric='F1-Score',
                       save_path='outputs/plots/classification_f1_comparison.png')
plot_model_comparison(clf_results_df, metric='Accuracy',
                       save_path='outputs/plots/classification_accuracy_comparison.png')
plot_confusion_matrices(trained_models, X_test, y_test,
                         save_path='outputs/plots/confusion_matrices.png')

if 'Random Forest' in trained_models:
    plot_feature_importance(trained_models['Random Forest'], feature_names,
                            'Random Forest', save_path='outputs/plots/feature_importance_rf.png')
if 'Gradient Boosting' in trained_models:
    plot_feature_importance(trained_models['Gradient Boosting'], feature_names,
                            'Gradient Boosting', save_path='outputs/plots/feature_importance_gb.png')

print("\n  Hyperparameter Tuning...")
best_rf, rf_params, rf_score = tune_random_forest(X_train, y_train)
best_gb, gb_params, gb_score = tune_gradient_boosting(X_train, y_train)
best_knn, knn_params, knn_score = tune_knn(X_train, y_train)

tuning_results = pd.DataFrame([
    {'Model': 'Random Forest', 'Best_Params': str(rf_params), 'Best_CV_F1': rf_score},
    {'Model': 'Gradient Boosting', 'Best_Params': str(gb_params), 'Best_CV_F1': gb_score},
    {'Model': 'KNN', 'Best_Params': str(knn_params), 'Best_CV_F1': knn_score},
])
tuning_results.to_csv('outputs/results/hyperparameter_tuning.csv', index=False)
print("  [OK] Classification complete")

# === STEP 5: REGRESSION ===
print("\n[STEP 5] Regression Analysis (7 models)")
X_reg_train, X_reg_test, y_reg_train, y_reg_test, reg_features = \
    prepare_regression_data(df_normalized, feature_cols=['Annual Income (k$)'])

reg_results_df, reg_trained = compare_regressors(
    X_reg_train, X_reg_test, y_reg_train, y_reg_test, poly_degree=3
)
reg_results_df.to_csv('outputs/results/regression_comparison.csv', index=False)

plot_regression_comparison(reg_results_df, save_path='outputs/plots/regression_r2_comparison.png')
plot_all_regressions_1d(reg_trained, X_reg_train, X_reg_test,
                         y_reg_train, y_reg_test,
                         save_path='outputs/plots/all_regression_lines.png')

best_reg_name = reg_results_df.iloc[0]['Model']
plot_regression(reg_trained[best_reg_name], X_reg_train, X_reg_test,
                y_reg_train, y_reg_test, best_reg_name,
                save_path='outputs/plots/best_regression_detail.png')

print("\n  Multi-Feature Regression...")
X_reg_train_m, X_reg_test_m, y_reg_train_m, y_reg_test_m, _ = \
    prepare_regression_data(df_normalized, feature_cols=['Age', 'Gender', 'Annual Income (k$)'])

reg_results_multi, _ = compare_regressors(
    X_reg_train_m, X_reg_test_m, y_reg_train_m, y_reg_test_m, poly_degree=2
)
reg_results_multi.to_csv('outputs/results/regression_multi_feature.csv', index=False)
print("  [OK] Regression complete")

# === STEP 6: APRIORI ===
print("\n[STEP 6] Association Rule Mining (Fixed)")
basket_binary = prepare_data(df_original)
freq_items_bin, rules_binary = apply_apriori(basket_binary, min_support=0.15, min_confidence=0.5)
print_top_rules(rules_binary, n=10)

basket_multi = prepare_data_multibin(df_original)
freq_items_multi, rules_multi = apply_apriori(basket_multi, min_support=0.1, min_confidence=0.4)
print_top_rules(rules_multi, n=10)

if len(rules_binary) > 0:
    rs = rules_binary.copy()
    rs['antecedents'] = rs['antecedents'].apply(lambda x: ', '.join(list(x)))
    rs['consequents'] = rs['consequents'].apply(lambda x: ', '.join(list(x)))
    rs.to_csv('outputs/results/association_rules_binary.csv', index=False)

if len(rules_multi) > 0:
    rm = rules_multi.copy()
    rm['antecedents'] = rm['antecedents'].apply(lambda x: ', '.join(list(x)))
    rm['consequents'] = rm['consequents'].apply(lambda x: ', '.join(list(x)))
    rm.to_csv('outputs/results/association_rules_multibin.csv', index=False)
print("  [OK] Apriori complete")

# === STEP 7: ADVANCED ANALYSIS ===
print("\n[STEP 7] Advanced Analysis")
X_all = df_normalized[['Age', 'Gender', 'Annual Income (k$)',
                         'Spending Score (1-100)']].values
pca_result, tsne_result = plot_pca_vs_tsne(
    X_all, labels=df_normalized['Cluster'].values,
    save_path='outputs/plots/pca_vs_tsne.png'
)
print("  PCA vs t-SNE: done")

outliers_iso, scores_iso = detect_outliers_isolation_forest(X_cluster_2d)
outliers_lof, scores_lof = detect_outliers_lof(X_cluster_2d)
plot_outliers(X_cluster_2d, outliers_iso, outliers_lof,
              xlabel="Income", ylabel="Spending",
              save_path='outputs/plots/outlier_detection.png')

both = outliers_iso & outliers_lof
print(f"  Outliers by BOTH methods: {both.sum()}")

print("  Running statistical significance tests...")
import sklearn.base
models_dict = get_all_models()
cv_raw_results = []
for name, model in models_dict.items():
    fresh = sklearn.base.clone(model)
    cv_res = cross_validate_model(fresh, X_full, y_full, cv=10, model_name=name)
    cv_raw_results.append(cv_res)

p_value_matrix = compare_all_pairs(cv_raw_results)
p_value_matrix.to_csv('outputs/results/pairwise_pvalues.csv')
print("  [OK] Advanced analysis complete")

# === STEP 8: SUMMARY ===
print("\n[STEP 8] Generating Summary Report")
report = generate_summary_report(
    clf_results=clf_results_df,
    reg_results=reg_results_df,
    cluster_results=cluster_results_df,
    save_path='outputs/reports/summary_report.txt'
)

print("\n" + "="*70)
print("  ALL DONE! Full pipeline completed successfully.")
print("="*70)
print(f"\n  Plots saved:   {len(os.listdir('outputs/plots'))} files")
print(f"  Results saved: {len(os.listdir('outputs/results'))} files")
print(f"  Reports saved: {len(os.listdir('outputs/reports'))} files")
print("="*70)
