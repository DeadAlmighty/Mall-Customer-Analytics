import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import our custom modules
from src.preprocessing import (
    load_data, clean_data, encode_gender, normalize_data,
    add_age_bins, add_income_category, add_spending_category
)
from src.clustering import compare_clustering, apply_kmeans
from src.models import prepare_classification_data, get_all_models
from src.regression import prepare_regression_data
from src.apriori import prepare_data, prepare_data_multibin, apply_apriori

# App configurations
st.set_page_config(
    page_title="Mall Customer Segment Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 800;
        color: #2E4057;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 18px;
        color: #566E7A;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2E4057;
        margin-bottom: 10px;
        color: #1A1A1A !important;
    }
    .metric-card h4, .metric-card h2, .metric-card p {
        color: #1A1A1A !important;
    }
    .cluster-card {
        background-color: #F0F4F8;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1070C0;
        margin-bottom: 10px;
        color: #1A1A1A !important;
    }
    .cluster-card h4, .cluster-card h2, .cluster-card p {
        color: #1A1A1A !important;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- Load & Process Data -----------------
@st.cache_data
def get_cached_data(filepath):
    df_raw = load_data(filepath)
    df_cleaned = clean_data(df_raw)
    
    # original state for EDA and Apriori
    df_orig = df_cleaned.copy()
    df_orig = add_age_bins(df_orig)
    df_orig = add_income_category(df_orig)
    df_orig = add_spending_category(df_orig)
    
    # encoded/normalized state for models
    df_enc = encode_gender(df_cleaned.copy())
    df_norm, scaler = normalize_data(df_enc, method='minmax')
    
    # Run K-means to assign clusters
    X_cluster = df_norm[['Annual Income (k$)', 'Spending Score (1-100)']].values
    labels, kmeans = apply_kmeans(X_cluster, n_clusters=5)
    
    df_norm['Cluster'] = labels
    df_orig['Cluster'] = labels
    
    return df_raw, df_orig, df_norm, scaler, kmeans

# Load default data
default_data_path = 'data/raw/mall_customers.csv'
if os.path.exists(default_data_path):
    df_raw, df_original, df_normalized, scaler, kmeans_model = get_cached_data(default_data_path)
else:
    st.error("Default dataset not found at 'data/raw/mall_customers.csv'. Please upload a file.")
    st.stop()

# Helper to display image safely
def show_plot(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Plot '{path}' not found. Please run the master pipeline to generate all plots.")

# ----------------- Sidebar Navigation -----------------
st.sidebar.image("https://img.icons8.com/clouds/200/brain.png", width=120)
st.sidebar.title("Mall Analytics Dashboard")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation Options",
    [
        "🏠 Overview & EDA",
        "🎯 Customer Clustering",
        "📊 Classification Models",
        "📈 Regression Analysis",
        "🔗 Association Rules",
        "🧪 Advanced Analysis",
        "🔮 Real-Time Predictions"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**DMW Capstone Project**  \nVersion 2.0 (Upgraded)")

# ----------------- 🏠 Overview & EDA -----------------
if menu == "🏠 Overview & EDA":
    st.markdown("<div class='main-title'>🧠 Customer Behavior Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Data Warehousing & Data Mining Capstone Project</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-card'><h4>Total Samples</h4><h2>982</h2><p>After data cleaning</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><h4>Average Age</h4><h2>38.9 yrs</h2><p>Range: 18 - 80 yrs</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><h4>Avg Annual Income</h4><h2>$57.1k</h2><p>Range: $15k - $144.1k</p></div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-card'><h4>Avg Spending Score</h4><h2>42.6</h2><p>Range: 1 - 92</p></div>", unsafe_allow_html=True)

    st.markdown("### 📊 Dataset Preview")
    tab_raw, tab_stats = st.tabs(["Raw Data (First 10 Rows)", "Statistical Overview"])
    with tab_raw:
        st.dataframe(df_raw.head(10), use_container_width=True)
    with tab_stats:
        st.dataframe(df_original.describe(), use_container_width=True)
        
    st.markdown("### 📈 Project EDA Plots (11 Plots Available)")
    tab_dist, tab_corr, tab_rel, tab_box = st.tabs([
        "Demographic Distributions", 
        "Correlation Heatmaps", 
        "Bivariate Relations", 
        "Outlier Boxplots"
    ])
    
    with tab_dist:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            show_plot('outputs/plots/age_distribution.png', "Age Distribution Curve")
            show_plot('outputs/plots/gender_distribution.png', "Gender Split (Male vs Female)")
        with col_d2:
            show_plot('outputs/plots/spending_distribution.png', "Spending Score Distribution")
            show_plot('outputs/plots/feature_distributions.png', "General Overview of Distributions")
            
    with tab_corr:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            show_plot('outputs/plots/correlation_matrix.png', "Global Feature Correlation Matrix")
        with col_c2:
            show_plot('outputs/plots/correlation_matrix_by_gender.png', "Feature Correlation Separated by Gender")
            
    with tab_rel:
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            show_plot('outputs/plots/income_vs_spending.png', "Income vs Spending Score Scatter plot")
            show_plot('outputs/plots/gender_vs_spending.png', "Gender vs Spending habits")
        with col_r2:
            show_plot('outputs/plots/age_vs_spending.png', "Age vs Spending Score")
            
    with tab_box:
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            show_plot('outputs/plots/boxplots.png', "Syllabus Standard Boxplots Overview")
        with col_b2:
            show_plot('outputs/plots/income_boxplot.png', "Annual Income Boxplot Details")

# ----------------- 🎯 Customer Clustering -----------------
elif menu == "🎯 Customer Clustering":
    st.markdown("<div class='main-title'>🎯 Customer Segmentation (Clustering)</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Analyzing segments using KMeans, DBSCAN, Hierarchical, and GMM</div>", unsafe_allow_html=True)
    
    st.markdown("### 📊 Clustering Metrics Comparison")
    if os.path.exists('outputs/results/clustering_comparison.csv'):
        compare_df = pd.read_csv('outputs/results/clustering_comparison.csv')
        st.table(compare_df)
    else:
        st.warning("Comparison metrics not found.")
        
    tab_models, tab_elbow = st.tabs(["Cluster Model Outputs", "Optimal Parameter Search"])
    
    with tab_models:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            show_plot('outputs/plots/clustering_comparison.png', "Comparison of 4 Methods side-by-side")
        with col_m2:
            show_plot('outputs/plots/kmeans_clusters.png', "K-Means Cluster Segmentation details")
            
    with tab_elbow:
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            show_plot('outputs/plots/elbow_method.png', "Elbow Method Curve (WCSS)")
            show_plot('outputs/plots/dendrogram.png', "Hierarchical Agglomerative Dendrogram")
        with col_e2:
            show_plot('outputs/plots/elbow_silhouette.png', "Elbow Method + Silhouette Scores Side-by-side")
            
    st.markdown("### 💡 Identified Customer Segments (KMeans)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='cluster-card'><h4>Cluster 0: Low Income - High Spending</h4><p>Target for value marketing, high propensity to buy but budget-conscious.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='cluster-card'><h4>Cluster 1: High Income - High Spending</h4><p>Premium category. Highly loyal & premium-tier promotions work best.</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='cluster-card'><h4>Cluster 2: Moderate Segment</h4><p>Balanced spending and income. Mainstream audience.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='cluster-card'><h4>Cluster 3: Low Income - Low Spending</h4><p>Frugal group. Only responsive to heavy discounts and absolute essentials.</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='cluster-card'><h4>Cluster 4: High Income - Low Spending</h4><p>High purchasing potential but conservative spending. Good target for upselling.</p></div>", unsafe_allow_html=True)

# ----------------- 📊 Classification Models -----------------
elif menu == "📊 Classification Models":
    st.markdown("<div class='main-title'>📊 Classification Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Predicting customer segment labels using 8 ML classifiers</div>", unsafe_allow_html=True)
    
    tab_res, tab_cv, tab_conf, tab_feat = st.tabs([
        "Test Set Performance", 
        "10-Fold Cross-Validation", 
        "Confusion Matrices", 
        "Feature Importance"
    ])
    
    with tab_res:
        if os.path.exists('outputs/results/classification_comparison.csv'):
            clf_df = pd.read_csv('outputs/results/classification_comparison.csv')
            st.dataframe(clf_df, use_container_width=True)
        else:
            st.warning("Classification results not found.")
            
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            show_plot('outputs/plots/classification_accuracy_comparison.png', "Classification Accuracy comparison")
        with col_c2:
            show_plot('outputs/plots/classification_f1_comparison.png', "Classification F1-Score comparison")
            
    with tab_cv:
        if os.path.exists('outputs/results/classification_cv_results.csv'):
            cv_df = pd.read_csv('outputs/results/classification_cv_results.csv')
            st.dataframe(cv_df, use_container_width=True)
        else:
            st.warning("CV results not found.")
            
    with tab_conf:
        show_plot('outputs/plots/confusion_matrices.png', "Multi-Model Confusion Matrices")
        
    with tab_feat:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            show_plot('outputs/plots/feature_importance_rf.png', "Random Forest Model Feature Importance")
        with col_f2:
            show_plot('outputs/plots/feature_importance_gb.png', "Gradient Boosting Model Feature Importance")

# ----------------- 📈 Regression Analysis -----------------
elif menu == "📈 Regression Analysis":
    st.markdown("<div class='main-title'>📈 Regression Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Modeling the relationships between demographics and spending trends</div>", unsafe_allow_html=True)
    
    tab_single, tab_multi = st.tabs(["Single Feature (Income -> Spending)", "Multi-Feature (Age + Gender + Income -> Spending)"])
    
    with tab_single:
        if os.path.exists('outputs/results/regression_comparison.csv'):
            reg_df = pd.read_csv('outputs/results/regression_comparison.csv')
            st.dataframe(reg_df, use_container_width=True)
        else:
            st.warning("Regression results not found.")
            
        col_reg1, col_reg2 = st.columns(2)
        with col_reg1:
            show_plot('outputs/plots/regression_plot.png', "Original Linear Regression trend line")
            show_plot('outputs/plots/best_regression_detail.png', "Residual plot details for top regression model")
        with col_reg2:
            show_plot('outputs/plots/all_regression_lines.png', "Comparative fits for all 7 regression models")
            
    with tab_multi:
        if os.path.exists('outputs/results/regression_multi_feature.csv'):
            reg_multi_df = pd.read_csv('outputs/results/regression_multi_feature.csv')
            st.dataframe(reg_multi_df, use_container_width=True)
        else:
            st.warning("Multi-feature regression results not found.")
            
        show_plot('outputs/plots/regression_r2_comparison.png', "Regression R-squared comparison chart")

# ----------------- 🔗 Association Rules -----------------
elif menu == "🔗 Association Rules":
    st.markdown("<div class='main-title'>🔗 Association Rule Mining (Apriori)</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Discovering behavioral rules using Median & Tercile splits</div>", unsafe_allow_html=True)
    
    st.info("The Apriori module was fixed to use median/quantile-based categorizations, eliminating false 99% confidence patterns.")
    
    tab_bin, tab_terc = st.tabs(["Median-based Binary Rules", "Tercile-based Multi-Bin Rules"])
    
    with tab_bin:
        if os.path.exists('outputs/results/association_rules_binary.csv'):
            rules_bin = pd.read_csv('outputs/results/association_rules_binary.csv')
            st.dataframe(rules_bin, use_container_width=True)
        else:
            st.warning("Binary association rules not found.")
            
    with tab_terc:
        if os.path.exists('outputs/results/association_rules_multibin.csv'):
            rules_terc = pd.read_csv('outputs/results/association_rules_multibin.csv')
            st.dataframe(rules_terc, use_container_width=True)
        else:
            st.warning("Tercile association rules not found.")

# ----------------- 🧪 Advanced Analysis -----------------
elif menu == "🧪 Advanced Analysis":
    st.markdown("<div class='main-title'>🧪 Advanced Analytical Techniques</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Showcasing dimensionality reduction, outlier detection, and statistical testing</div>", unsafe_allow_html=True)
    
    tab_dim, tab_out, tab_stat = st.tabs([
        "PCA vs t-SNE Projections", 
        "Outlier Detection", 
        "Statistical Paired t-tests"
    ])
    
    with tab_dim:
        show_plot('outputs/plots/pca_vs_tsne.png', "PCA vs t-SNE side-by-side 2D cluster projection")
        show_plot('outputs/plots/pca_plot.png', "PCA dimension coordinates scatter")
        
    with tab_out:
        show_plot('outputs/plots/outlier_detection.png', "Anomalies filtered by Isolation Forest vs Local Outlier Factor")
        
    with tab_stat:
        st.markdown("### 🧪 Paired t-Test Significance Matrix (p-values)")
        if os.path.exists('outputs/results/pairwise_pvalues.csv'):
            p_val_df = pd.read_csv('outputs/results/pairwise_pvalues.csv', index_col=0)
            st.dataframe(p_val_df.style.background_gradient(cmap="coolwarm", axis=None), use_container_width=True)
        else:
            st.warning("Statistical significance matrix not found.")

# ----------------- 🔮 Real-Time Predictions -----------------
elif menu == "🔮 Real-Time Predictions":
    st.markdown("<div class='main-title'>🔮 Real-Time Segment Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Enter customer demographics to dynamically classify their cluster segment</div>", unsafe_allow_html=True)
    
    # Gather inputs
    st.markdown("#### Enter Customer Profile:")
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        age_in = st.slider("Customer Age", min_value=18, max_value=80, value=35)
        gender_in = st.selectbox("Gender", ["Female", "Male"])
    with col_in2:
        income_in = st.slider("Annual Income (k$)", min_value=15.0, max_value=145.0, value=60.0, step=0.5)
        spending_in = st.slider("Spending Score (1-100)", min_value=1.0, max_value=100.0, value=50.0, step=1.0)
        
    if st.button("🔮 Predict Customer Segment"):
        # Normalize the custom inputs
        gender_enc = 1 if gender_in == "Male" else 0
        raw_vals = np.array([[gender_enc, age_in, income_in, spending_in]])
        
        # Fit scaler on variables as done in preprocessing
        features_norm = df_normalized[['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)']].values
        
        # Build training set to run predictor
        X_train, X_test, y_train, y_test, _ = prepare_classification_data(df_normalized, target_col='Cluster')
        
        # Get random forest classifier to run live predictions
        from sklearn.ensemble import RandomForestClassifier
        rf_clf = RandomForestClassifier(max_depth=10, min_samples_split=5, n_estimators=100, random_state=42)
        rf_clf.fit(X_train, y_train)
        
        # Normalization calculation
        # Normalized values in df_normalized match original scaled ranges
        min_age, max_age = df_original['Age'].min(), df_original['Age'].max()
        min_inc, max_inc = df_original['Annual Income (k$)'].min(), df_original['Annual Income (k$)'].max()
        min_spd, max_spd = df_original['Spending Score (1-100)'].min(), df_original['Spending Score (1-100)'].max()
        
        scaled_age = (age_in - min_age) / (max_age - min_age)
        scaled_inc = (income_in - min_inc) / (max_inc - min_inc)
        scaled_spd = (spending_in - min_spd) / (max_spd - min_spd)
        
        scaled_input = np.array([[scaled_age, gender_enc, scaled_inc, scaled_spd]])
        
        predicted_cluster = rf_clf.predict(scaled_input)[0]
        probabilities = rf_clf.predict_proba(scaled_input)[0]
        
        st.markdown("---")
        st.success(f"### Predicted Cluster Segment: **Cluster {predicted_cluster}**")
        
        cluster_descs = {
            0: "📉 **Low Income - Low Spending (Frugal)**",
            1: "🛍️ **Low Income - High Spending (Aspirational)**",
            2: "⚖️ **Moderate Segment (Average Spending)**",
            3: "💰 **High Income - High Spending (Premium / Elite)**",
            4: "🏦 **High Income - Low Spending (Conservative Saver)**"
        }
        
        st.markdown(f"#### Profile Description: {cluster_descs.get(predicted_cluster, 'Unknown')}")
        
        # Display probabilities
        prob_df = pd.DataFrame({
            "Cluster": [f"Cluster {i}" for i in range(5)],
            "Description": [cluster_descs[i] for i in range(5)],
            "Confidence Probability": [f"{p*100:.2f}%" for p in probabilities]
        })
        st.table(prob_df)
