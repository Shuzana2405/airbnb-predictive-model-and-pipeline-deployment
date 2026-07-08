# BTT
# Airbnb Predictive Model & Pipeline Deployment

An end-to-end machine learning project focused on building, optimizing, and deploying a binary classification pipeline to predict customer target labels from complex Airbnb listing profiles.

## 🚀 Project Overview
This project walks through a complete machine learning lifecycle, from structural feature engineering to persistence and deployment readiness. The primary goal is to maximize prediction accuracy (AUC-ROC) utilizing optimized logistic regression architectures.

## 🛠️ Tech Stack & Key Libraries
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Visualization:** Seaborn, Matplotlib
* **Model Persistence:** Pickle

## 📈 Key Results & Methodology
* **Hyperparameter Tuning:** Automated structural model tuning using `GridSearchCV` with 5-fold cross-validation to pinpoint the optimal regularization strength ($C$).
* **Optimized Metrics:** Reached a final test Area Under the Curve (AUC) score of **0.816** by thoroughly evaluating trade-offs across Precision-Recall and ROC curves.
* **Feature Selection:** Utilized `SelectKBest` with ANOVA F-values (`f_classif`) to isolate highly predictive metrics and benchmark dimensional resource trade-offs.
* **Persistence:** Implemented secure object serialization via `pickle` to output a reusable binary `model_best.pkl` artifact ready for pipeline integration.

## 📂 Repository Structure
* `ModelSelectionForLogisticRegression.ipynb` — Core development notebook containing pipeline logic, training operations, and diagnostic visualization.
* `airbnbData_train.csv` — Feature-engineered training and validation listing profiles.
* `model_best.pkl` — Packaged, high-performance binary model output.
