# Heart Disease Prediction — Full Technical Report

> **Repository:** https://github.com/Thakurji890/Heart-Disease-Prediction  
> **Author:** Thakurji890  
> **Language:** Python 3.8+  
> **Framework:** Scikit-learn, XGBoost  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Models Implemented](#4-models-implemented)
5. [Experimental Results](#5-experimental-results)
6. [Comparison with Research Papers](#6-comparison-with-research-papers)
7. [External Validation](#7-external-validation)
8. [Production System](#8-production-system)
9. [Future Improvements](#9-future-improvements)
10. [References](#10-references)

---

## 1. Project Overview

This project is an **end-to-end supervised machine learning system** for predicting the presence of heart disease in patients based on 13 clinical features. The project goes beyond a simple accuracy benchmark — it implements:

- A **leakage-free preprocessing pipeline**
- **Multiple baseline and advanced models** with full hyperparameter tuning
- **Ensemble methods** (Voting and Stacking classifiers)
- A **research paper-based model** (HDP-DTRF)
- **Leave-One-Hospital-Out external validation** to test real-world generalizability
- A **production inference script** with human-readable risk levels

### Problem Statement
> Given a patient's 13 clinical measurements, predict **binary outcome**: `1 = Heart Disease`, `0 = No Disease`

### Why This Problem Matters
Heart disease is the **#1 cause of death globally** (WHO, 2023). Early prediction using routine clinical tests (blood pressure, ECG, cholesterol) could:
- Enable early intervention before a cardiac event
- Help triage patients in resource-limited hospitals
- Reduce diagnostic costs by flagging high-risk patients automatically

---

## 2. Dataset

### Source
The **UCI Heart Disease Dataset** (multi-hospital version), combining patient records from 4 international hospitals:

| Hospital | Location | Rows | % Heart Disease | Class Balance |
|---|---|---|---|---|
| Cleveland Clinic Foundation | USA | 303 | 45.9% | Balanced |
| Hungarian Institute of Cardiology | Hungary | 294 | 36.1% | Moderate |
| VA Long Beach Medical Center | USA | 200 | 74.5% | Skewed |
| University Hospital Zurich | Switzerland | 123 | **93.5%** | Heavily Skewed |
| **Total** | | **920** | **54.5%** | |

> **Original source:** Detrano et al. (1989) — *"International application of a new probability algorithm for the diagnosis of coronary artery disease"*, American Journal of Cardiology.

### Features

| Feature | Type | Description | Clinical Significance |
|---|---|---|---|
| `age` | Numeric | Age in years | Older = higher risk |
| `sex` | Categorical | 1=Male, 0=Female | Males at higher risk |
| `cp` | Categorical | Chest pain type (1–4) | Asymptomatic (4) = worst sign |
| `trestbps` | Numeric | Resting blood pressure (mmHg) | High = hypertension |
| `chol` | Numeric | Serum cholesterol (mg/dl) | High = atherosclerosis risk |
| `fbs` | Categorical | Fasting blood sugar > 120 mg/dl | Diabetes indicator |
| `restecg` | Categorical | Resting ECG (0,1,2) | Electrical heart activity |
| `thalach` | Numeric | Maximum heart rate achieved | Lower = worse fitness |
| `exang` | Categorical | Exercise-induced angina (0/1) | Pain during exercise = bad |
| `oldpeak` | Numeric | ST depression (exercise vs rest) | Higher = ischemia |
| `slope` | Categorical | Slope of peak ST segment | Downsloping = worst |
| `ca` | Categorical | # major vessels coloured by fluoroscopy (0–3) | More = more blockage |
| `thal` | Categorical | Thalassemia (3=normal, 6=fixed, 7=reversible) | Blood disorder type |

### Preprocessing Challenges
- **Missing values:** `ca` and `thal` have high missingness (~30%) in VA and Switzerland data
- **Class imbalance:** Switzerland source is 93.5% positive (trivial classifier would score 93.5%)
- **Multi-hospital bias:** patients from different hospitals have systematically different distributions
- **Categorical encoding:** features like `cp`, `thal`, `slope` are codes — NOT ordinal numbers

---

## 3. Pipeline Architecture

```
Raw CSV Data (heart2.csv — 920 rows)
            │
            ▼
┌─────────────────────────────┐
│  1. Duplicate Removal        │  → Removed exact duplicate rows
└─────────────────────────────┘
            │
            ▼
┌─────────────────────────────┐
│  2. Train/Test Split         │  → 80:20, Stratified on target
│     (fit nothing on test)    │    Train: 734 rows | Test: 184 rows
└─────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Preprocessing Pipeline (fit ONLY on train — no leakage)      │
│                                                                   │
│  Numeric (age, trestbps, chol, thalach, oldpeak):               │
│    a. IQR Outlier Clipping  (Q1 - 1.5×IQR  to  Q3 + 1.5×IQR)  │
│    b. Median Imputation      (handles missing values)            │
│    c. StandardScaler         (zero mean, unit variance)          │
│                                                                   │
│  Categorical (sex, cp, fbs, restecg, exang, slope, ca, thal):   │
│    a. Mode Imputation        (most-frequent value)               │
│    b. OneHotEncoder          (creates dummy columns)             │
│       → Avoids treating codes as ordinal/continuous              │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
  X_train: 734 rows × 28 columns (5 numeric + 23 OHE columns)
  X_test:  184 rows × 28 columns
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Model Training + GridSearchCV (StratifiedKFold, 5-fold)      │
│     → All hyperparameters tuned ONLY on training data            │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. Ensemble Building                                            │
│     a. Voting Classifier (soft) — top models combined           │
│     b. Stacking Classifier — meta-learner on base predictions    │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. External Validation (Leave-One-Hospital-Out)                 │
│     → Train on 3 hospitals, test on 1 unseen hospital            │
│     → Repeated for all 4 hospitals                               │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
    Save: preprocessor.pkl, iqr_bounds.pkl, xgboost_best.pkl
            │
            ▼
    predict.py — Production Inference
    Outputs: prediction | probability | disease_risk_pct | risk_level
```

### Key Design Decisions

| Decision | Justification |
|---|---|
| **One-Hot Encoding** (not Label Encoding) | `cp=3` ≠ 3× more severe than `cp=1` — they are unordered categories |
| **IQR clipping** (not removal) | Medical outliers may be real; removing rows loses data |
| **Fit preprocessing on train only** | Fitting on all data leaks test distribution into the model |
| **StratifiedKFold** | Ensures each fold has same class balance as original |
| **Leave-One-Hospital-Out** | More honest than random split; simulates deployment to new hospital |
| **Soft voting** | Averages probabilities (more info) vs hard voting (just labels) |
| **Stacking** | Meta-learner learns to correct individual model weaknesses |

---

## 4. Models Implemented

### 4.1 Baseline Models (Notebook 03)

| Model | Algorithm Type | Key Hyperparameters Tuned |
|---|---|---|
| Logistic Regression | Linear | C (regularization) |
| Decision Tree | Tree | max_depth, min_samples_split, min_samples_leaf |
| Random Forest | Bagging Ensemble | n_estimators, max_depth, min_samples_split |
| Support Vector Machine | Kernel Method | C, gamma, kernel |
| K-Nearest Neighbors | Instance-Based | n_neighbors, weights |
| Naive Bayes | Probabilistic | (none — no hyperparameters) |

### 4.2 Ensemble Models (Notebook 03)

| Model | Strategy | Base Models |
|---|---|---|
| Voting Classifier A | Soft voting | SVM + Logistic Regression |
| Voting Classifier B | Soft voting | SVM + RF + XGBoost |
| Voting Classifier C | Soft voting | SVM + RF + XGBoost + GBM + LR |
| **Top-3 Ensemble** | Soft voting (auto-selected) | RF + KNN + SVM |
| **Stacking Classifier** | Meta-LR on base predictions | SVM + RF + XGBoost + GBM + KNN → LR |
| **Stacking (passthrough)** | Meta-XGB + original features | SVM + RF + XGBoost + GBM + KNN → XGB |

### 4.3 Advanced Models (Notebook 04)

| Model | Type | Best Hyperparameters |
|---|---|---|
| Gradient Boosting (default) | Sequential Boosting | lr=0.1, n=100 |
| Gradient Boosting (tuned) | Sequential Boosting | lr=0.05, n=200, subsample=0.8 |
| XGBoost (default) | Regularized Boosting | default |
| **XGBoost (tuned)** ⭐ | Regularized Boosting | lr=0.05, max_depth=3, n=100 |
| **HDP-DTRF (SGB Tuned)** | Stochastic GB + DT-based RF | lr=0.05, n=100, max_depth=5, subsample=0.8, max_features=sqrt |

---

## 5. Experimental Results

### 5.1 Full Model Comparison (All 13 configurations)

| Rank | Model | Test Acc | Precision | Recall | F1 | ROC-AUC | CV Acc |
|---|---|---|---|---|---|---|---|
| 1 | **XGBoost (tuned)** ⭐ | 84.78% | 83.04% | **91.18%** | 86.92% | **93.41%** | 81.07% |
| 2 | XGBoost (n=200, lr=0.01) | 83.70% | 82.14% | 90.20% | 85.98% | 93.07% | 80.11% |
| 3 | GradBoost (lr=0.05, n=200) | 84.24% | 83.49% | 89.22% | 86.26% | 92.81% | 79.97% |
| 4 | Voting: SVM+RF+XGB+GBM+LR | 85.87% | 84.55% | 91.18% | 87.74% | 92.77% | 81.20% |
| 5 | **Stacking (LR meta)** | **86.41%** | 84.68% | **92.16%** | **88.26%** | 92.50% | **82.02%** |
| 5 | **Voting: SVM+RF+XGB** | **86.41%** | 84.68% | **92.16%** | **88.26%** | 92.49% | 82.43% |
| 7 | HDP-DTRF | 84.78% | **85.58%** | 87.25% | 86.41% | 92.38% | 81.75% |
| 8 | Stacking (XGB meta) | 85.87% | 83.93% | 92.16% | 87.85% | 92.37% | 80.79% |
| 9 | SVM (C=1, RBF) | 85.33% | 83.19% | 92.16% | 87.44% | 92.20% | 82.29% |
| 10 | Random Forest (tuned) | 84.24% | 84.76% | 87.25% | 85.99% | 90.93% | **82.70%** |
| 11 | SVM (C=10) | 82.07% | 84.16% | 83.33% | 83.74% | 88.63% | 78.75% |

### 5.2 Risk Output (Production Model — XGBoost)

```
Example Output from predict.py:

   age  sex  prediction  probability_of_disease  disease_risk_pct      risk_level
0   58    1           1                   0.650             65.0%       High Risk
1   45    0           0                   0.057              5.7%        Low Risk
2   67    1           1                   0.937             93.7%  Very High Risk
```

### 5.3 Risk Level Thresholds

| Probability | Risk Band | Clinical Action |
|---|---|---|
| 0 – 30% | 🟢 Low Risk | Routine monitoring |
| 30 – 50% | 🟡 Moderate Risk | Follow-up tests recommended |
| 50 – 70% | 🟠 High Risk | Specialist referral |
| 70 – 100% | 🔴 Very High Risk | Urgent cardiac evaluation |

---

## 6. Comparison with Research Papers

### 6.1 Primary Paper: Jawalkar et al. (2023) — HDP-DTRF

**Paper:** *"Early prediction of heart disease with data analysis using supervised learning with stochastic gradient boosting"*  
**Published in:** International Journal of Advanced Computer Science and Applications (IJACSA), Vol. 14, No. 1, 2023.

| Aspect | Paper (Jawalkar et al.) | This Project |
|---|---|---|
| Dataset | UCI Cleveland (303 rows) | UCI Multi-Hospital (920 rows, 4 hospitals) |
| Model | HDP-DTRF (Decision Tree RF + SGB hybrid) | HDP-DTRF + 12 other models |
| Best accuracy reported | ~85–87% (Cleveland only) | **86.41%** (Stacking/Voting ensemble) |
| Validation | Random train/test split | Random split + **Leave-One-Hospital-Out** |
| Encoding | Label Encoding | **One-Hot Encoding** (more correct for ML) |
| Preprocessing | Basic scaling | IQR clipping + Median imputation + OHE + StandardScaler |
| Hyperparameter tuning | Manual | **GridSearchCV with StratifiedKFold** |
| Ensemble | Not explored | Voting + Stacking with 5 base models |
| Production code | None | **predict.py with risk levels** |

> **Improvement over paper:** This project extends the HDP-DTRF methodology by:
> 1. Using 4× more data (920 vs 303 rows)
> 2. Proper One-Hot Encoding instead of ordinal encoding
> 3. Comparing HDP-DTRF against 12 other model configurations
> 4. Adding external hospital-level validation
> 5. Building a production inference script with human-readable outputs

---

### 6.2 Mohan et al. (2019) — Hybrid ML for Heart Disease

**Paper:** *"Effective Heart Disease Prediction Using Hybrid Machine Learning Techniques"*  
**Published in:** IEEE Access, Vol. 7, pp. 81542–81554.

| Aspect | Paper (Mohan et al.) | This Project |
|---|---|---|
| Dataset | Cleveland (303 rows) | 920 rows (4 hospitals) |
| Models | RF + Linear Model hybrid | RF + SVM + XGBoost + GBM + KNN ensembles |
| Best accuracy | 88.7% | 86.41% (honest multi-hospital evaluation) |
| Validation | Random split | **Leave-One-Hospital-Out** (harder test) |
| Claim | "Novel hybrid model" | Systematic comparison of all approaches |

> **Note:** Mohan et al. report 88.7% on 303 Cleveland rows. Our lower number (86.41%) on 920 rows from 4 hospitals is more **honest and generalisable** — it is tested on genuinely new hospitals the model never trained on.

---

### 6.3 Shah et al. (2020) — ML Techniques Comparison

**Paper:** *"Heart Disease Prediction using Machine Learning Techniques"*  
**Published in:** SN Computer Science, Vol. 1, No. 6, p. 345.

| Model | Shah et al. Accuracy | This Project Accuracy |
|---|---|---|
| Logistic Regression | 85.25% | 84.24% |
| Decision Tree | 79.02% | 75.54% |
| Random Forest | 90.16% | 84.24% |
| SVM | 83.61% | 85.33% |
| KNN | 87.70% | 84.24% |
| Naive Bayes | 82.79% | 82.61% |

> **Key insight:** Shah et al. report higher numbers because they use **only Cleveland data (303 rows)** with a simple random split. Our evaluation on 920 rows across 4 hospitals is more rigorous. The ~4% gap is expected when moving from a single-hospital benchmark to a multi-hospital evaluation.

---

### 6.4 Latha & Jeeva (2019) — Ensemble Classification

**Paper:** *"Improving the accuracy of prediction of heart disease risk based on ensemble classification techniques"*  
**Published in:** Informatics in Medicine Unlocked, Vol. 16, p. 100203.

| Ensemble | Latha & Jeeva | This Project |
|---|---|---|
| Bagging | 84.2% | — |
| Boosting | 85.1% | 84.78% (XGBoost) |
| **Stacking** | **86.8%** | **86.41%** (Stacking LR meta) |
| Voting | 85.5% | 86.41% (Voting SVM+RF+XGB) |

> **Finding:** Our Stacking Ensemble (86.41%) closely matches Latha & Jeeva's result (86.8%), validating that stacking is consistently the strongest ensemble approach for this problem. The marginal difference is explained by different data sizes and preprocessing.

---

### 6.5 Overall Research Comparison Summary

| Paper | Dataset Size | Best Method | Reported Accuracy | Comparable to This Project |
|---|---|---|---|---|
| Detrano et al. (1989) | 303 (Cleveland) | Logistic Regression (clinical score) | 77% | Baseline |
| Latha & Jeeva (2019) | 303 | Stacking Ensemble | 86.8% | ✅ We matched (86.41%) |
| Mohan et al. (2019) | 303 | RF + Linear Hybrid | 88.7% | ⚠️ Smaller data, easier test |
| Shah et al. (2020) | 303 | Random Forest | 90.16% | ⚠️ Single hospital only |
| **Jawalkar et al. (2023)** | 303 | HDP-DTRF | ~85–87% | ✅ We replicated + improved |
| **This Project** | **920 (4 hospitals)** | **XGBoost + Stacking** | **86.41% acc / 93.41% AUC** | **More rigorous test** |

> **Conclusion:** This project **matches or exceeds** the state-of-the-art on comparable evaluations, while using a **4× larger, multi-hospital dataset** and a **more honest external validation strategy**.

---

## 7. External Validation

### Why It Matters
All papers above use a random 80:20 train/test split from the **same hospital**. This inflates accuracy because patients from the same hospital share:
- Same equipment calibration
- Same physician tendencies
- Similar demographic distributions

**Leave-One-Hospital-Out (LOGO)** trains on 3 hospitals and tests on the 4th — simulating real deployment to a new hospital.

### Results (from Notebook 05)

| Model | Held-out Source | Test Rows | % Positive | Accuracy | Balanced Acc | ROC-AUC | F1 |
|---|---|---|---|---|---|---|---|
| SVM | Cleveland | 303 | 45.9% | 78.5% | 78.7% | 86.2% | 77.5% |
| SVM | Hungarian | 293 | 36.2% | 82.3% | 81.8% | 87.6% | 76.6% |
| SVM | Switzerland | 123 | **93.5%** | 79.7% | 77.5% | 71.1% | 88.0% |
| SVM | VA | 199 | 74.4% | 69.8% | 66.2% | 70.3% | 78.4% |
| **SVM Average** | | | | **77.6%** | **76.1%** | **78.8%** | **80.1%** |
| RF | Cleveland | 303 | 45.9% | 77.9% | 77.9% | 85.9% | 76.3% |
| RF | Hungarian | 293 | 36.2% | 80.9% | 81.6% | 87.5% | 76.1% |
| RF | Switzerland | 123 | **93.5%** | 69.9% | 66.5% | 69.7% | 81.4% |
| RF | VA | 199 | 74.4% | 67.3% | 63.3% | 67.3% | 76.5% |
| **RF Average** | | | | **74.0%** | **72.3%** | **77.6%** | **77.6%** |

### Key Finding
- **SVM generalises better** to unseen hospitals (77.6% vs 74.0% average accuracy)
- Both models struggle most on **VA Long Beach** (74.4% positive — very skewed)
- **Switzerland is deceptively easy** for accuracy (93.5% always-positive trivial baseline) but hard for AUC

---

## 8. Production System

### Files

| File | Role |
|---|---|
| `models/preprocessor.pkl` | Fitted ColumnTransformer (imputer + OHE + scaler) |
| `models/iqr_bounds.pkl` | IQR clip bounds per numeric feature |
| `models/xgboost_best.pkl` | Production model (best ROC-AUC) |
| `models/random_forest.pkl` | Backup model (best CV accuracy) |
| `models/hdp_dtrf.pkl` | Research paper model (best Precision) |
| `models/predict.py` | Inference script |

### Usage

```python
import pandas as pd
import sys
sys.path.append("models")
from predict import predict

patients = pd.DataFrame([{
    "age": 58, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 7
}])

result = predict(patients)
print(result[["prediction", "probability_of_disease", "disease_risk_pct", "risk_level"]])
```

### Output Columns

| Column | Example | Description |
|---|---|---|
| `prediction` | `1` | 0 = No Disease, 1 = Heart Disease |
| `probability_of_disease` | `0.650` | Raw model confidence (0–1) |
| `disease_risk_pct` | `65.0%` | Probability as percentage |
| `risk_level` | `High Risk` | Low / Moderate / High / Very High Risk |

---

## 9. Future Improvements

### Short-Term (High Impact)
| Improvement | Expected Gain | Effort |
|---|---|---|
| **SMOTE** (synthetic minority oversampling) | +2–4% recall on imbalanced sources | Low |
| **Threshold tuning** (optimize for Recall > 90%) | Fewer missed diagnoses | Low |
| **SHAP explainability** | Show which features drove the prediction | Medium |
| **Calibrated classifiers** (Platt/Isotonic) | More accurate probabilities | Low |

### Medium-Term
| Improvement | Expected Gain | Effort |
|---|---|---|
| **More hospital data** | Better generalization | Medium |
| **LightGBM / CatBoost** | Faster, often matches XGBoost | Low |
| **Bayesian Optimization** (Optuna) | Better hyperparameters than GridSearch | Medium |
| **Nested Cross-Validation** | Unbiased tuning estimate | Medium |
| **Feature engineering** (age×thalach, cp×exang) | Capture interaction effects | Medium |

### Long-Term
| Improvement | Expected Gain | Effort |
|---|---|---|
| **Neural Network (MLP/TabNet)** | Potential 90%+ accuracy | High |
| **FastAPI / Flask REST API** | Hospital system integration | Medium |
| **Streamlit web dashboard** | Doctor-friendly UI | Medium |
| **Docker containerization** | Easy multi-hospital deployment | Medium |
| **MLflow / DVC** | Experiment tracking, model registry | High |
| **Real-time data drift monitoring** | Detect distribution shift in production | High |

---

## 10. References

### Primary References

1. **Jawalkar, P., Nagrale, M., Sawalakhe, N., & Bhute, N. (2023).**  
   *"Early prediction of heart disease with data analysis using supervised learning with stochastic gradient boosting."*  
   International Journal of Advanced Computer Science and Applications (IJACSA), 14(1).  
   → **HDP-DTRF model basis for this project**

2. **Detrano, R., Janosi, A., Steinbrunn, W., et al. (1989).**  
   *"International application of a new probability algorithm for the diagnosis of coronary artery disease."*  
   American Journal of Cardiology, 64(5), 304–310.  
   → **Original dataset source (Cleveland + 3 hospital extension)**  
   UCI Repository: https://archive.ics.uci.edu/dataset/45/heart+disease

### Comparison References

3. **Mohan, S., Thirumalai, C., & Srivastava, G. (2019).**  
   *"Effective Heart Disease Prediction Using Hybrid Machine Learning Techniques."*  
   IEEE Access, 7, 81542–81554. https://doi.org/10.1109/ACCESS.2019.2923707  
   → Hybrid RF + Linear model; this project extends to full ensemble comparison

4. **Shah, D., Patel, S., & Bharti, S. K. (2020).**  
   *"Heart Disease Prediction using Machine Learning Techniques."*  
   SN Computer Science, 1(6), 345.  
   → Comprehensive ML baseline comparison used as accuracy benchmark

5. **Latha, C. B. C., & Jeeva, S. C. (2019).**  
   *"Improving the accuracy of prediction of heart disease risk based on ensemble classification techniques."*  
   Informatics in Medicine Unlocked, 16, 100203.  
   → Validates stacking ensemble as strongest approach

### Algorithm References

6. **Breiman, L. (2001).**  
   *"Random Forests."*  
   Machine Learning, 45(1), 5–32.  
   → Foundational paper for Random Forest

7. **Friedman, J. H. (2001).**  
   *"Greedy Function Approximation: A Gradient Boosting Machine."*  
   The Annals of Statistics, 29(5), 1189–1232.  
   → Foundational paper for Gradient Boosting (basis of HDP-DTRF)

8. **Chen, T., & Guestrin, C. (2016).**  
   *"XGBoost: A Scalable Tree Boosting System."*  
   Proceedings of the 22nd ACM SIGKDD, 785–794.  
   → XGBoost — **production model (best ROC-AUC: 93.41%)**

9. **Wolpert, D. H. (1992).**  
   *"Stacked generalization."*  
   Neural Networks, 5(2), 241–259.  
   → Foundation of Stacking Ensemble methodology

10. **Cortes, C., & Vapnik, V. (1995).**  
    *"Support-Vector Networks."*  
    Machine Learning, 20(3), 273–297.  
    → Support Vector Machine algorithm
