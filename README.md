# 🧠 DMW Project – Customer Behavior Analysis  
### *Data Mining & Data Warehousing Capstone*

---

## 📌 Overview

**Dataset:** Mall Customer Segmentation Dataset  
**Tools Used:** Python, Scikit-learn, Streamlit, Jupyter  

This project implements a complete **Data Mining + Data Warehouse pipeline** to analyze mall customer behavior and generate actionable business insights.

It integrates:
- Clustering (Customer Segmentation using KMeans, DBSCAN, Hierarchical, GMM)
- Classification (8 Prediction Models with Cross-Validation and Tuning)
- Regression (7 Trend Analysis models in 1D and Multi-feature)
- Association Rule Mining (Fixed Apriori)
- Advanced Analysis (t-SNE, Outlier Detection, Statistical paired t-tests)
- Data Warehouse + OLAP (Decision Support Star Schema)
- Interactive Streamlit Dashboard

---

## 🎯 Objectives

- Segment customers based on income & spending  
- Predict customer categories using ML models with high accuracy (96%+)
- Analyze demographic spending trends using linear/SVR trends
- Discover valid behavior patterns using robust Apriori association rules  
- Build Star Schema Data Warehouse  
- Perform interactive OLAP slice/dice/roll-up actions  

---

## 📂 Project Structure

```
DMW_Project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_pca.ipynb
│   ├── 04_kmeans.ipynb
│   ├── 05_classification.ipynb
│   ├── 06_regression.ipynb
│   ├── 07_apriori.ipynb
│   ├── 08_warehouse_olap.ipynb
│   ├── 09_main_pipeline.ipynb
│   ├── 10_prediction.ipynb
│   └── 11_advanced_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── clustering.py
│   ├── models.py
│   ├── regression.py
│   ├── apriori.py
│   └── advanced_analysis.py
│
├── outputs/
│   ├── plots/
│   ├── results/
│   └── reports/
│
├── app.py
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

# 📊 Key Results

## 🔹 Classification Performance (Test Set)

| Model | Accuracy | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|
| **MLP Neural Net** | **96.45%** | **0.9645** | **0.9965** |
| Random Forest | 95.94% | 0.9593 | 0.9989 |
| Gradient Boosting | 95.94% | 0.9591 | 0.9936 |
| Decision Tree | 95.43% | 0.9541 | 0.9710 |
| Naive Bayes | 94.92% | 0.9491 | 0.9939 |
| KNN | 79.19% | 0.7908 | 0.9512 |
| Logistic Regression | 75.63% | 0.7553 | 0.9408 |
| SVM | 57.87% | 0.5161 | 0.8833 |

✔ MLP Neural Net and Random Forest achieve near-perfect segmentation prediction.  
✔ 10-Fold Cross-Validation confirms high generalization performance with low variance.

---

## 🔹 Regression Analysis (Spending Trend)

| Model | R² (1D) | R² (Multi-Feature) | RMSE (Multi) |
|:---|:---:|:---:|:---:|
| **SVR** | 0.6858 | **0.7149 (Best)** | **0.1069** |
| **Polynomial (deg=2)** | **0.6876 (Best)** | 0.7123 | 0.1073 |
| Random Forest | 0.6602 | 0.7061 | 0.1085 |
| Ridge | 0.6776 | 0.6959 | 0.1104 |
| Linear | 0.6728 | 0.6947 | 0.1106 |

✔ Single Feature (Income -> Spending): Polynomial (deg=3) is the best trend line (R² = 0.6876).  
✔ Multi-Feature (Age + Gender + Income -> Spending): SVR achieves the highest accuracy (R² = 0.7149).

---

## 🔹 Association Rules (Fixed Apriori)

| Rule | Confidence | Lift |
|:---|:---:|:---:|
| High Income → Low Spending, Senior | 76.6% | 1.91 |
| Low Income → High Spending, Young | 73.1% | 1.87 |
| Age Senior → Income High | **100%** | **2.94** |

✔ Fixed the previous threshold bug to avoid data leakage.  
✔ Discovered statistically valid demographic associations.

---

## 🔹 Clustering Insights (2D & 4D)

- Optimal Clusters: **5 (Elbow Method & Silhouette validation)**  
- Methods evaluated: **KMeans, DBSCAN, Agglomerative Hierarchical, GMM**

| Cluster | Description | Marketing Strategy |
|:---:|:---|:---|
| 0 | Low Income – Low Spending | Heavy discounts, value essentials |
| 1 | Low Income – High Spending | Aspirational offers, impulse buys |
| 2 | Medium Segment | Mainstream advertising, loyalty cards |
| 3 | High Income – Low Spending | Up-selling high-value items |
| 4 | High Income – High Spending | Premium tier, exclusive invites, personalized treatment |

---

# 📈 Visualizations

All pre-generated pipeline outputs are saved in `outputs/plots/`:
- Customer Segmentation: `outputs/plots/clustering_comparison.png`
- Elbow & Silhouette Curves: `outputs/plots/elbow_silhouette.png`
- Hierarchical Dendrogram: `outputs/plots/dendrogram.png`
- Multi-Model Confusion Matrices: `outputs/plots/confusion_matrices.png`
- Feature Importance: `outputs/plots/feature_importance_rf.png`
- Regression Fits: `outputs/plots/all_regression_lines.png`
- PCA vs t-SNE Projections: `outputs/plots/pca_vs_tsne.png`
- Outlier Detection (Isolation Forest vs LOF): `outputs/plots/outlier_detection.png`

---

# 🔄 Data Pipeline

```
Raw Data
   ↓
Preprocessing
   ↓
EDA
   ↓
PCA
   ↓
Clustering
   ↓
Classification
   ↓
Regression
   ↓
Apriori
   ↓
Data Warehouse + OLAP
```

---

# 🏗️ Data Warehouse Design

## ⭐ Star Schema

- **Fact Table:**  
  - Spending Score  
  - Annual Income  

- **Dimension Tables:**  
  - Age  
  - Gender  
  - Income Category  

---

## 🔍 OLAP Operations

- Roll-up → Aggregation  
- Drill-down → Detailed analysis  
- Slice → Filter one dimension  
- Dice → Multi-condition filtering  

---

# ⚙️ Tech Stack

- Python  
- Streamlit (Interactive Dashboard)  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- MLxtend  
- Jupyter Notebook  

---

# 🚀 How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Launch the Dashboard (Streamlit App)
```bash
streamlit run app.py
```

### Run the Notebooks in Order:
You can also run the full pipeline notebook directly:
```
notebooks/11_advanced_analysis.ipynb
```
Or explore individual steps:
```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10
```

---

# 🧠 Key Insights

- **5 Distinct Customer Segments** identified using KMeans and validated via Silhouette/Davies-Bouldin metrics.
- **High predictive power**: Multi-Layer Perceptron (MLP) and Random Forest classifiers achieve **96.45% accuracy** in predicting customer segments.
- **Demographic spending trends**: Support Vector Regression (SVR) and Polynomial Regression show strong fitting when predicting spending score from Age + Income.
- **Apriori Association Rules**: Strong associations show distinct cluster-based habits (e.g. older high-income buyers are conservative savers).
- **Outliers**: Isolation Forest and Local Outlier Factor successfully filter ~5% anomalies.

---

# 🏆 Project Highlights

✔ Covers ALL Capstone Syllabus Modules  
✔ Industry-standard Modular Architecture (`src/`)  
✔ Multi-model Comparison (8 Classifiers, 7 Regressors, 4 Clustering methods)  
✔ Statistical Significance Validation (Paired t-tests)  
✔ Data Warehouse + OLAP Slice/Dice/Roll-up  
✔ Complete interactive Streamlit dashboard  

---

# 🔮 Future Improvements

- Integrate real-time API streaming datasets
- Build a personalized product recommendation engine for each segment
- Expand customer dimension features with loyalty tier history data
