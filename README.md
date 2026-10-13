<div align="center">

# ⚙️ Scania APS Sensor Fault Prediction (`SENSOR`)
### End-to-End Modular MLOps Pipeline for Heavy-Vehicle Air Pressure System Diagnostics

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=for-the-badge)](https://catboost.ai/)
[![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![AWS S3](https://img.shields.io/badge/AWS_S3-569A31?style=for-the-badge&logo=amazons3&logoColor=white)](https://aws.amazon.com/s3/)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/features/actions)

*A production-grade Machine Learning & MLOps system engineered to detect Air Pressure System (APS) component failures in Scania heavy-duty trucks—minimizing asymmetric fleet maintenance costs through automated telemetry cleaning, SMOTEENN resampling, cost-sensitive classification, and AWS S3 model deployment.*

<p align="center">
  <a href="#-problem-statement--cost-matrix">Problem Statement</a> •
  <a href="#-dataset--exploratory-data-analysis">Dataset & EDA</a> •
  <a href="#-experimental-benchmarking--model-selection">Benchmarks & Leaderboard</a> •
  <a href="#-system-architecture--flowcharts">Architecture</a> •
  <a href="#-mlops-pipeline-components">Pipeline Stages</a> •
  <a href="#-project-structure">Project Structure</a> •
  <a href="#-getting-started">Quick Start</a>
</p>

</div>

---

## 📌 Problem Statement & Cost Matrix

The **Air Pressure System (APS)** generates pressurized air utilized across critical heavy-truck functions such as braking and gear shifting. In fleet diagnostics:
- **Positive Class (`pos`):** Component failures originating specifically within the APS subsystem.
- **Negative Class (`neg`):** Trucks experiencing failures in components unrelated to the APS.

Misclassifications carry a highly asymmetric financial penalty:

| True \ Predicted | Predicted: `Positive` (APS Fault) | Predicted: `Negative` (No APS Fault) |
| :--- | :---: | :---: |
| **True: `Positive`** | True Positive (`$0`) | **False Negative (`Cost_2 = $500`)** |
| **True: `Negative`** | **False Positive (`Cost_1 = $10`)** | True Negative (`$0`) |

```text
Total_Cost = (10 × FP) + (500 × FN)
```

- **`Cost_1 ($10 — Type I Error / False Positive):`** Cost of an unnecessary preventative mechanic inspection at a workshop.
- **`Cost_2 ($500 — Type II Error / False Negative):`** Cost of missing a faulty APS truck, leading to an unplanned roadside breakdown (**50× more expensive** than a false alarm).

**Core Objective:** Minimize `Total_Cost` by aggressively suppressing False Negatives while maintaining **>99% resampled test accuracy (>96% raw holdout accuracy)**.

---

## 📊 Dataset & Exploratory Data Analysis

Explored in [`Notebook/Scania_APS_failure_prediction.ipynb`](Notebook/Scania_APS_failure_prediction.ipynb):

- **Raw Dimensions:** `36,188 rows × 171 columns`
- **Feature Composition:** `170` anonymized numerical telemetry features (`aa_000` through `eg_000` representing counters and histogram bins) + `1` categorical target column (`class`).
- **Severe Class Imbalance (~35:1 Ratio):**
  - `neg` (Majority Class): **35,188 instances** (97.24%)
  - `pos` (Minority Class): **1,000 instances** (2.76%)

### Missing Value Profiling & Feature Pruning
Almost all sensor columns contain missing values encoded as `"na"`. Columns exceeding a **70% missingness threshold** are systematically dropped:

| Dropped Feature | Missing Percentage (%) |
| :--- | :---: |
| `br_000` | `81.41%` |
| `bq_000` | `80.50%` |
| `bp_000` | `78.79%` |
| `ab_000` | `77.09%` |
| `cr_000` | `77.09%` |
| `bo_000` | `76.53%` |
| `bn_000` | `72.76%` |

- **Post-Pruning Dimensions:** `36,188 rows × 164 columns`
- **Remaining Missing Cells:** **5.37%** of total dataset cells, handled downstream via `SimpleImputer` / `KNNImputer` and `RobustScaler`.

---

## 🧪 Experimental Benchmarking & Model Selection

To identify the optimal preprocessing and classification architecture before building the production `SENSOR` package, **5 distinct feature-engineering experiments** were conducted across **8 machine learning classifiers** using `RobustScaler` (to handle extreme skewness/outliers) and **SMOTEENN** (hybrid Synthetic Minority Over-sampling + Edited Nearest Neighbors to clean overlapping class boundaries).

### 1. Imputation & Dimensionality Reduction Experiments

| Experiment | Imputation / Feature Strategy | Preprocessing Pipeline | Best Classifier | Test Accuracy | Total Misclassification Cost (`$`) |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **Exp 1** | **KNN Imputer** (`n_neighbors=3`) | `RobustScaler` + `KNNImputer` + `SMOTEENN` | `XGBClassifier` | `99.57%` | `$2,950` |
| **Exp 2** | **Median Imputation** | `RobustScaler` + `SimpleImputer(median)` + `SMOTEENN` | `XGBClassifier` | `99.64%` | `$2,680` |
| **Exp 3** | **MICE Imputation** (`IterativeImputer`) | `RobustScaler` + `MICE` + `SMOTEENN` | `XGBClassifier` | `99.55%` | `$3,220` |
| **Exp 4** | **Constant Imputation (`fill_value=0`)** | `RobustScaler` + `SimpleImputer(constant=0)` + `SMOTEENN` | `XGBClassifier` | **`99.66%`** | **`$2,450`** |
| **Exp 5** | **PCA Dimensionality Reduction** | `RobustScaler` + `SimpleImputer(0)` + `PCA` + `SMOTEENN` | `RandomForest` | `98.40%` | `$11,150` |

> **Key Engineering Takeaway:** Sensor telemetry missingness (`na`) often indicates a sensor or subsystem was inactive (`0` count) rather than randomly corrupted. Consequently, **Constant Imputation (`fill_value=0`)** paired with `RobustScaler` and `SMOTEENN` achieved the lowest operational penalty (`$2,450`) and highest recall on faulty trucks, while running orders of magnitude faster than `KNNImputer` and `MICE`.

### 2. Classifier Leaderboard (Best Experiment — Constant Imputation + SMOTEENN)

| Rank | Model Name | Test Accuracy | F1-Score | Precision | Recall (Sensitivity) | ROC-AUC | Total Cost (`10×FP + 500×FN`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **XGBoost (`XGBClassifier`)** | **99.66%** | **0.9966** | **0.9942** | **0.9990** | **0.9966** | **$2,450** |
| 🥈 **2** | **CatBoost (`CatBoostClassifier`)** | 99.59% | 0.9959 | 0.9931 | 0.9987 | 0.9959 | $2,980 |
| 🥉 **3** | **Random Forest (`RandomForestClassifier`)** | 99.41% | 0.9941 | 0.9905 | 0.9977 | 0.9941 | $4,660 |
| **4** | **Gradient Boosting (`GradientBoostingClassifier`)** | 98.54% | 0.9855 | 0.9813 | 0.9897 | 0.9854 | $9,410 |
| **5** | **Decision Tree (`DecisionTreeClassifier`)** | 98.62% | 0.9862 | 0.9834 | 0.9891 | 0.9862 | $9,870 |
| **6** | **K-Nearest Neighbors (`KNeighborsClassifier`)** | 97.65% | 0.9768 | 0.9672 | 0.9866 | 0.9765 | $11,920 |
| **7** | **AdaBoost (`AdaBoostClassifier`)** | 96.21% | 0.9624 | 0.9558 | 0.9691 | 0.9621 | $24,650 |
| **8** | **Logistic Regression (`LogisticRegression`)** | 93.40% | 0.9351 | 0.9210 | 0.9496 | 0.9340 | $41,380 |

---

## 🏗️ System Architecture & Flowcharts

### Overall Process Architecture
<p align="center">
  <img src="Flowcharts/Process.png" alt="Overall Process Architecture" width="85%"/>
</p>

<details>
<summary><b>📡 View Streaming Data Producer & Consumer Diagrams (Draw.io)</b></summary>
<br>
<p align="center">
  <b>Data Producer Workflow</b><br>
  <img src="docs/producer.drawio%20(2).svg" alt="Producer Architecture" width="80%"/>
</p>
<p align="center">
  <b>Data Consumer Workflow</b><br>
  <img src="docs/consumer.drawio%20(2).svg" alt="Consumer Architecture" width="80%"/>
</p>
</details>

### End-to-End Training & Prediction Pipelines
<p align="center">
  <img src="Flowcharts/0_Sensor%20Training%20Pipeline.png" alt="Sensor Training Pipeline" width="85%"/>
</p>
<p align="center">
  <img src="Flowcharts/Sensor%20Prediction%20Pipeline.png" alt="Sensor Prediction Pipeline" width="85%"/>
</p>

---

## 🧩 MLOps Pipeline Components

The production package (`SENSOR/pipeline/training_pipeline.py`) orchestrates six modular stages governed by immutable configuration (`config_entity.py`) and output artifact (`artifact_entity.py`) dataclasses:

### 1. Data Ingestion (`SENSOR/components/data_ingestion.py`)
- Queries raw sensor telemetry from **MongoDB Atlas** via `SENSOR/data_access/sensor_data.py`.
- Replaces `"na"` string tokens with `np.nan`, drops MongoDB `_id` metadata, persists a snapshot to the local Feature Store, and executes a stratified train-test split.

<p align="center">
  <img src="Flowcharts/Sensor_Data%20Ingestion%20Component.png" alt="Data Ingestion Component" width="75%"/>
</p>

### 2. Data Validation (`SENSOR/components/data_validation.py`)
- Validates train and test DataFrames against [`config/schema.yaml`](config/schema.yaml) (checking column counts, numerical types, and mandatory drop columns).
- Runs two-sample **Kolmogorov-Smirnov (KS) Data Drift** tests (`scipy.stats.ks_2samp`) across all numerical features and exports a YAML drift report.

<p align="center">
  <img src="Flowcharts/Sensor_Data%20Validation%20Component%20-Photoroom.png" alt="Data Validation Component" width="75%"/>
</p>

### 3. Data Transformation (`SENSOR/components/data_transformation.py`)
- Constructs a Scikit-learn `Pipeline` combining `RobustScaler` (mitigating extreme sensor outliers) and `SimpleImputer(strategy="constant", fill_value=0)`.
- Encodes target classes (`neg` → `0`, `pos` → `1`) via `TargetValueMapping` and applies **SMOTEENN** hybrid resampling (`imbalanced-learn`) to synthesize minority fault instances and clean noisy decision boundaries.
- Serializes the preprocessor (`preprocessing.pkl`) and transformed `.npy` arrays.

<p align="center">
  <img src="Flowcharts/Sensor_Data_Transformation_Component-Photoroom.png" alt="Data Transformation Component" width="75%"/>
</p>

### 4. Model Trainer (`SENSOR/components/model_trainer.py`)
- Fits the winning gradient-boosted classifier (`XGBClassifier`) on the resampled training matrix.
- Computes `ClassificationMetricArtifact` (`f1_score`, `precision_score`, `recall_score`), verifying that model performance exceeds the configured expected accuracy threshold and remains within overfitting tolerances.
- Packages the fitted preprocessor and trained estimator into a unified `SensorModel` inference object (`SENSOR/ml/metric/model/estimator.py`).

<p align="center">
  <img src="Flowcharts/Sensor_Model%20Trainer%20Component.png" alt="Model Trainer Component" width="75%"/>
</p>

### 5. Model Evaluation (`SENSOR/components/model_evaluation.py`)
- Benchmarks the newly trained Challenger model against the existing Champion production model stored in `saved_models/`.
- Approves model promotion only if the Challenger improves F1 performance beyond `MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE`.

<p align="center">
  <img src="Flowcharts/Sensor%20Model%20Evaluation%20Component.png" alt="Model Evaluation Component" width="75%"/>
</p>

### 6. Model Pusher (`SENSOR/components/model_pusher.py`)
- Version-stamps the accepted `SensorModel` into the production model registry and synchronizes artifacts and models with **Amazon S3** via `SENSOR/cloud_storage/s3_syncer.py`.

<p align="center">
  <img src="Flowcharts/Sensor%20Model%20Pusher%20Component.png" alt="Model Pusher Component" width="75%"/>
</p>

---

## 🛠️ Tech Stack

| Layer | Technologies & Libraries |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Machine Learning** | `scikit-learn`, `xgboost`, `catboost`, `imbalanced-learn` (SMOTEENN), `scipy` |
| **Data & Visualization** | `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly` |
| **Database & Cloud** | `pymongo` (MongoDB Atlas), `boto3` & AWS CLI (Amazon S3) |
| **Config & Packaging** | `PyYAML` (`schema.yaml`), `dill`, `python-dotenv`, `setuptools` |
| **CI/CD & DevOps** | GitHub Actions, Docker |

---

## 📂 Project Structure

```text
APS---Sensor/
├── config/
│   └── schema.yaml                         # Column definitions, numerical types & drop list
├── docs/
│   ├── producer.drawio (2).svg             # Telemetry producer architecture diagram
│   └── consumer.drawio (2).svg             # Telemetry consumer architecture diagram
├── Flowcharts/                             # Architectural & component-level MLOps diagrams
│   ├── 0_Sensor Training Pipeline.png
│   ├── Process.png
│   ├── Sensor_Data Ingestion Component.png
│   ├── Sensor_Data Validation Component -Photoroom.png
│   ├── Sensor_Data_Transformation_Component-Photoroom.png
│   ├── Sensor_Model Trainer Component.png
│   ├── Sensor Model Evaluation Component.png
│   ├── Sensor Model Pusher Component.png
│   └── Sensor Prediction Pipeline.png
├── github/workflow/
│   └── main.yaml                           # CI/CD pipeline configuration
├── Notebook/
│   └── Scania_APS_failure_prediction.ipynb # Full EDA, 5 imputation trials & 8-model benchmark
├── SENSOR/                                 # Installable MLOps Python package
│   ├── __init__.py
│   ├── config.py                           # Environment & client configuration
│   ├── exception.py                        # Custom traceback exception handler
│   ├── logger.py                           # Timestamped execution logger
│   ├── utils2.py                           # Auxiliary helper functions
│   ├── cloud_storage/
│   │   ├── s3_storage.py                   # Boto3 S3 storage interface
│   │   └── s3_syncer.py                    # AWS CLI bucket synchronization utility
│   ├── components/
│   │   ├── data_ingestion.py               # Stage 1: MongoDB extraction & train/test split
│   │   ├── data_validation.py              # Stage 2: Schema validation & KS drift detection
│   │   ├── data_transformation.py          # Stage 3: RobustScaler, Imputer & SMOTEENN
│   │   ├── model_trainer.py                # Stage 4: Model training & SensorModel bundling
│   │   ├── model_evaluation.py             # Stage 5: Champion vs. Challenger evaluation
│   │   └── model_pusher.py                 # Stage 6: Model registry promotion
│   ├── configuration/
│   │   └── mongodb_db_connection.py        # MongoDB Atlas connection singleton
│   ├── constant/
│   │   ├── application.py                  # App host/port constants
│   │   ├── database.py                     # Database & collection identifiers
│   │   ├── env_variable.py                 # Environment variable keys
│   │   ├── s3_bucket.py                    # S3 bucket names
│   │   └── training_pipeline/__init__.py   # Pipeline file names, thresholds & constants
│   ├── data_access/
│   │   └── sensor_data.py                  # MongoDB-to-DataFrame data access layer
│   ├── entity/
│   │   ├── config_entity.py                # Pipeline stage configuration entities
│   │   └── artifact_entity.py              # Pipeline stage output artifact entities
│   ├── ml/metric/
│   │   ├── classification_metric.py        # F1, Precision & Recall metric evaluator
│   │   └── model/estimator.py              # TargetValueMapping, SensorModel & ModelResolver
│   ├── pipeline/
│   │   └── training_pipeline.py            # End-to-end pipeline orchestrator
│   └── utils/
│       └── main_utils.py                   # YAML, NumPy & Dill object serialization utilities
├── logs/                                   # Timestamped runtime execution logs
├── setup.py                                # Package setup specification
├── requirements.txt                        # Project dependencies
└── README.md                               # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/AbhyanandSharma2005/APS---Sensor.git](https://github.com/AbhyanandSharma2005/APS---Sensor.git)
cd APS---Sensor
```

### 2. Create & Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Configure Environment Variables
Create a `.env` file in the project root directory with your MongoDB Atlas and AWS credentials:
```env
MONGO_DB_URL="mongodb+srv://<username>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority"
AWS_ACCESS_KEY_ID="<your-aws-access-key>"
AWS_SECRET_ACCESS_KEY="<your-aws-secret-key>"
AWS_DEFAULT_REGION="us-east-1"
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 5. Run the Project
- **Run the Production Training Pipeline:**
  ```bash
  python main.py
  ```
- **Explore the Research & Benchmarking Notebook:**
  ```bash
  jupyter notebook Notebook/Scania_APS_failure_prediction.ipynb
  ```

---

## 📝 Logging & Exception Handling

- **Structured Execution Logs (`SENSOR/logger.py`):** Every pipeline execution creates a timestamped `.log` file inside `logs/` recording stage transitions, drift p-values, and classification metrics.
- **Custom Exception Tracebacks (`SENSOR/exception.py`):** Captures the exact Python module filename, line number, and system error traceback across all pipeline components.

---

## 👤 Author

**Abhyanand Sharma**
- **GitHub:** [@AbhyanandSharma2005](https://github.com/AbhyanandSharma2005)
- **Repository:** [APS---Sensor](https://github.com/AbhyanandSharma2005/APS---Sensor)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.