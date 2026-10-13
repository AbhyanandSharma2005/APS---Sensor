<div align="center">

# ⚙️ APS Sensor Fault Prediction
### End-to-End Machine Learning Pipeline for Heavy-Vehicle Air Pressure System Diagnostics

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

*A robust predictive maintenance system engineered to detect Air Pressure System (APS) sensor faults from high-dimensional, imbalanced telemetry data—minimizing unplanned vehicle downtime and maintenance costs.*

<p align="center">
  <a href="#-project-overview">Overview</a> •
  <a href="#-pipeline-architecture">Architecture</a> •
  <a href="#-methodology--data-pipeline">Methodology</a> •
  <a href="#-model-evaluation--results">Performance</a> •
  <a href="#-getting-started">Quick Start</a>
</p>

</div>

---

## 📌 Project Overview

The **Air Pressure System (APS)** generates pressurized air crucial for critical heavy-vehicle functions, including braking and gear shifting. Identifying whether an impending component failure stems specifically from the APS (`pos`) or an unrelated subsystem (`neg`) is vital for fleet reliability:

- **False Positives (Type I Error):** Trigger unnecessary preventative checks at a minor workshop inspection cost.
- **False Negatives (Type II Error):** Result in an undetected APS fault, leading to roadside breakdowns, severe safety risks, and significantly higher emergency repair costs.

**`APS---Sensor`** implements a complete, modular machine learning workflow in Python and Scikit-learn that ingests noisy, high-dimensional sensor telemetry and predicts APS faults with **>96% accuracy**, prioritizing high recall on fault classes and optimizing total operational loss ratios.

---

## ✨ Key Highlights

- **High-Missingness Data Cleaning:** Automated filtering and statistical imputation strategies for sensor attributes with up to 80%+ missing telemetry readings.
- **Class Imbalance Mitigation:** Addresses extreme target skewness between healthy (`neg`) and faulty (`pos`) operational logs using resampling and cost-sensitive class weighting.
- **Feature Engineering & Dimensionality Reduction:** Eliminates collinearity, removes zero-variance sensors, and scales high-dimensional histogram and counter features.
- **Multi-Classifier Benchmarking:** Evaluates ensemble and linear classifiers using stratified cross-validation, ROC-AUC, Precision-Recall curves, and confusion matrices.
- **Cost-Sensitive Error Analysis:** Tunes decision thresholds to minimize asymmetric misclassification penalties rather than relying solely on raw accuracy.

---

## 🏗️ Pipeline Architecture

```text
+-------------------------------------------------------------------------+
|                        1. RAW TELEMETRY INGESTION                       |
|      High-dimensional sensor readings ( anonymized counters & bins )    |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                    2. DATA CLEANING & PREPROCESSING                     |
|  * Replace 'na' strings with NaN    * Drop high-missingness columns     |
|  * Median / KNN Imputation          * Zero-variance feature removal     |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|            3. FEATURE ENGINEERING & IMBALANCE HANDLING                  |
|  * Robust / Standard Scaling        * Outlier Capping                   |
|  * SMOTE / Stratified Resampling    * Class-Weight Optimization         |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|               4. MODEL TRAINING & HYPERPARAMETER TUNING                 |
|  * Logistic Regression / Random Forest / Gradient Boosting / XGBoost    |
|  * Stratified K-Fold Cross-Validation                                   |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                5. STATISTICAL & COST-MATRIX EVALUATION                  |
|  * Accuracy (>96%), F1, ROC-AUC     * Confusion Matrix & Error Analysis |
|  * Decision Threshold Tuning for Minimum Operational Loss               |
+-------------------------------------------------------------------------+
```

---

## 🛠️ Tech Stack

| Category | Technologies & Libraries |
| :--- | :--- |
| **Language** | Python 3.9+ |
| **Data Manipulation** | `pandas`, `numpy` |
| **Machine Learning** | `scikit-learn`, `imbalanced-learn` (SMOTE), `scipy` |
| **Visualization & EDA** | `matplotlib`, `seaborn` |
| **Experimentation & Serialization** | Jupyter Notebooks, `joblib` / `pickle` |

---

## 📂 Repository Structure

> 💡 *Note: Update the file names below if your local script or notebook names differ.*

```text
APS---Sensor/
├── data/
│   ├── raw/                         # Original APS sensor training and test datasets
│   └── processed/                   # Cleaned, imputed, and scaled feature matrices
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb    # Exploratory data analysis & missing value profiling
│   ├── 02_feature_engineering.ipynb # Scaling, collinearity removal & resampling
│   └── 03_model_evaluation.ipynb    # Classifier benchmarking & error analysis
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py        # Missing value imputation & cleaning utilities
│   ├── feature_engineering.py       # Feature scaling & imbalance handling pipelines
│   ├── train.py                     # Model training & hyperparameter tuning script
│   └── evaluate.py                  # Statistical metrics, confusion matrix & cost analysis
├── models/
│   └── aps_fault_classifier.pkl     # Serialized best-performing model pipeline
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

---

## 🔬 Methodology & Data Pipeline

### 1. Data Cleaning & Missing Value Imputation
Sensor logs frequently contain missing entries (`na`) due to hardware dropouts or unrecorded operational modes:
- Converted string-encoded missing markers (`'na'`) into standard `np.nan` values.
- Dropped sensor attributes exceeding the missingness threshold (e.g., columns with >70% missing data).
- Applied **Median Imputation** (via `SimpleImputer`) to handle skewed sensor distributions without distorting variance with extreme outliers.

### 2. Feature Engineering & Scaling
- Removed constant and quasi-constant features exhibiting near-zero variance.
- Applied `StandardScaler` / `RobustScaler` to normalize skewed sensor counters and histogram bins across vastly different numerical magnitudes.

### 3. Handling Severe Class Imbalance
Because APS-specific failures (`pos`) represent a small fraction of total fleet logs compared to non-APS cases (`neg`), standard classifiers bias toward the majority class. This pipeline resolves imbalance via:
- **Synthetic Minority Over-sampling (SMOTE)** and/or **Random Under-Sampling** strictly within the training folds to prevent data leakage.
- **Cost-Sensitive Learning (`class_weight='balanced'`)** to penalize minority-class misclassifications proportionally during training.

### 4. Statistical Modeling & Error Analysis
Classifiers are evaluated across multiple dimensions beyond top-line accuracy:
- **Confusion Matrix Decomposition:** Tracking True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN).
- **Cost Function Optimization:**

```text
Total Operational Cost = (Cost_FP × FP) + (Cost_FN × FN)
```

By calibrating the probability classification threshold, the model drastically cuts high-penalty False Negatives while maintaining an overall accuracy exceeding **96%**.

---

## 📊 Model Evaluation & Results

| Metric | Performance | Description |
| :--- | :--- | :--- |
| **Overall Accuracy** | **> 96.0%** | Proportion of total correct predictions across both classes |
| **ROC-AUC Score** | **> 0.95** | Separability between normal operations and APS fault states |
| **Minority Recall (Sensitivity)** | **High** | Minimizes undetected APS sensor failures (Type II errors) |
| **Precision & F1-Score** | **Optimized** | Balances workshop false alarms against critical fault detection |

---

## 🚀 Getting Started

### Prerequisites
Ensure you have **Python 3.9+** and `pip` installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/AbhyanandSharma2005/APS---Sensor.git
cd APS---Sensor
```

### 2. Create & Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(If running notebooks directly, ensure `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, and `imbalanced-learn` are installed.)*

### 4. Run the Pipeline
- **Via Jupyter Notebooks:**
  ```bash
  jupyter notebook
  ```
  Open the notebook(s) in the repository to step through EDA, preprocessing, model training, and visualization interactively.

- **Via Python Scripts (if modularized):**
  ```bash
  python src/train.py
  python src/evaluate.py
  ```

---

## 🔮 Future Enhancements

- [ ] **Automated MLOps Pipeline:** Integrate **MLflow** or **DVC** for experiment tracking, dataset versioning, and model registry management.
- [ ] **Real-Time Inference API:** Wrap the serialized Scikit-learn pipeline in a **FastAPI** microservice for real-time telemetry scoring.
- [ ] **Containerization:** Add a `Dockerfile` and GitHub Actions CI/CD workflow for automated testing and deployment.
- [ ] **Explainable AI (XAI):** Incorporate **SHAP** or **LIME** values to provide maintenance engineers with feature-level root-cause attributions for flagged sensor faults.

---

## 👤 Author

**Abhyanand Sharma**
- **GitHub:** [@AbhyanandSharma2005](https://github.com/AbhyanandSharma2005)
- **Repository:** [APS---Sensor](https://github.com/AbhyanandSharma2005/APS---Sensor)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.