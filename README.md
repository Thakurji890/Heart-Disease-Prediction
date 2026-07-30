# Heart Disease Prediction

An end-to-end machine learning project for predicting heart disease using clinical patient data. The project covers the full ML pipeline — from exploratory data analysis to hyperparameter tuning, advanced ensemble methods, and rigorous external validation across multiple hospital sources.

> **Repository:** https://github.com/Thakurji890/Heart-Disease-Prediction

---

## Project Structure

```
Heart-Disease-Prediction/
├── data/
│   ├── raw/                   # Original datasets (heart.csv, heart2.csv, etc.)
│   └── processed/             # Preprocessed train/test splits (CSV)
├── models/
│   ├── preprocessor.pkl       # Saved ColumnTransformer pipeline
│   ├── random_forest.pkl      # Best tuned baseline model
│   ├── hdp_dtrf.pkl           # HDP-DTRF (Gradient Boosting) model
│   └── predict.py             # Production inference script
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Baseline_Models.ipynb
│   ├── 04_All_Models_Comparison.ipynb
│   ├── 05_External_Validation.ipynb
│   └── 06_Ensemble_And_Evaluation_Honesty.ipynb
├── requirement.txt
├── steps.txt
└── README.md
```

---

## Dataset

The project uses the **UCI Heart Disease Dataset** (multi-hospital version) combining patient records from 4 hospitals:

| Source | Rows | % Positive (Heart Disease) |
|---|---|---|
| Cleveland Clinic | 303 | 45.9% |
| Hungarian Institute | 294 | 36.1% |
| VA Long Beach | 200 | 74.5% |
| University Hospital Zurich (Switzerland) | 123 | 93.5% |

**13 clinical features used:**
`age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (cholesterol), `fbs` (fasting blood sugar), `restecg` (resting ECG), `thalach` (max heart rate), `exang` (exercise-induced angina), `oldpeak` (ST depression), `slope`, `ca` (major vessels), `thal`

---

## Notebooks Overview

### 1. `01_EDA.ipynb` — Exploratory Data Analysis
- Dataset loading and initial inspection
- Shape, column types, and statistical summaries
- Missing values and duplicate record checks
- Target class distribution
- Univariate analysis (histograms, boxplots)
- Bivariate analysis and correlation heatmap
- Initial research observations

### 2. `02_Preprocessing.ipynb` — Data Preprocessing & Feature Engineering
- Missing value imputation (median for numeric, mode for categorical)
- Duplicate record removal
- Outlier treatment using the IQR method (clip strategy, fit on train only)
- **One-Hot Encoding** for categorical features (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`) to avoid ordinal bias
- Feature scaling with `StandardScaler` (fit on train only — no data leakage)
- Train/test split (80:20 with stratification)
- Feature importance analysis
- Preprocessor pipeline saved as `models/preprocessor.pkl`

### 3. `03_Baseline_Models.ipynb` — Baseline Model Training & Ensemble
- Training 6 baseline classifiers:
  - Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes
- Additional **Voting Classifier (SVM + Logistic Regression)** using soft voting
- **`GridSearchCV` hyperparameter tuning** applied to every tunable model using `StratifiedKFold`
- Automatic selection of the overall best model by CV accuracy
- **Top-3 Ensemble**: A soft-voting `VotingClassifier` built from the 3 best tuned models
- Confusion matrix, classification report, and ROC curve for the best model
- Best model saved as `models/random_forest.pkl`

  | Model | Best CV Accuracy |
  |---|---|
  | **Random Forest** | **0.827** ✅ |
  | Ensemble (RF + KNN + SVM) | 0.824 |
  | KNN | 0.824 |
  | Support Vector Machine | 0.823 |
  | Voting Classifier (SVM + LR) | 0.823 |
  | Logistic Regression | 0.808 |
  | Naive Bayes | 0.755 |
  | Decision Tree | 0.749 |

### 4. `04_All_Models_Comparison.ipynb` — Advanced Model Comparison
- Gradient Boosting and XGBoost comparison
- **HDP-DTRF model** (Decision Tree-based Random Forest + Stochastic Gradient Boosting)
  - Based on the methodology from *"Early prediction of heart disease with data analysis using supervised learning with stochastic gradient boosting"* (Jawalkar et al., 2023)
  - Uses `GradientBoostingClassifier` with `subsample=0.8` and `max_features="sqrt"` for stochastic behaviour
- Saved as `models/hdp_dtrf.pkl`

### 5. `05_External_Validation.ipynb` — Leave-One-Hospital-Out Validation
- Tests how well models **generalize to unseen hospitals**, not just unseen patients
- Uses **Leave-One-Group-Out** strategy across 4 hospital sources:
  - Cleveland, Hungarian, VA Long Beach, Switzerland
- Trains on 3 hospitals, tests on the 4th — rotates through all 4
- Reports Accuracy, **Balanced Accuracy**, **ROC-AUC**, and F1 (to handle class imbalance in Switzerland/VA sources)
- Compares SVM vs. Random Forest on cross-hospital generalization

### 6. `06_Ensemble_And_Evaluation_Honesty.ipynb` — Ensemble & Evaluation Honesty
- Further ensemble experiments and evaluation integrity checks
- Honest evaluation using held-out test sets

---

## Requirements

Python 3.8+ is required. Install all dependencies with:

```bash
pip install -r requirement.txt
```

### Dependencies
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `joblib`
- `nbformat`

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Thakurji890/Heart-Disease-Prediction.git
   cd Heart-Disease-Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```

3. **Run notebooks in order**
   ```
   01_EDA.ipynb
   02_Preprocessing.ipynb
   03_Baseline_Models.ipynb
   04_All_Models_Comparison.ipynb
   05_External_Validation.ipynb
   06_Ensemble_And_Evaluation_Honesty.ipynb
   ```

4. **Run inference on new patients**
   ```bash
   python models/predict.py
   ```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| One-Hot Encoding for categoricals | Avoids treating ordinal codes as continuous values |
| IQR clipping fit only on train | Prevents data leakage from test statistics |
| StratifiedKFold for CV | Preserves class balance in every fold |
| Leave-One-Hospital-Out validation | More honest real-world generalization test |
| Soft-voting ensemble | Averages probability scores for better calibrated decisions |
| HDP-DTRF model | Based on published research for heart disease prediction |

---

## References & Research Papers

### Primary Reference (Model Methodology)
1. **Jawalkar, P., Nagrale, M., Sawalakhe, N., & Bhute, N. (2023).**
   *"Early prediction of heart disease with data analysis using supervised learning with stochastic gradient boosting."*
   International Journal of Advanced Computer Science and Applications (IJACSA), 14(1).
   → Basis for the **HDP-DTRF hybrid model** implemented in this project.

### Dataset
2. **Detrano, R., Janosi, A., Steinbrunn, W., et al. (1989).**
   *"International application of a new probability algorithm for the diagnosis of coronary artery disease."*
   American Journal of Cardiology, 64(5), 304–310.
   → Original source of the **Cleveland, Hungarian, VA, and Switzerland** heart disease datasets.
   UCI Repository: https://archive.ics.uci.edu/dataset/45/heart+disease

### Related Works
3. **Shah, D., Patel, S., & Bharti, S. K. (2020).**
   *"Heart Disease Prediction using Machine Learning Techniques."*
   SN Computer Science, 1(6), 345.
   → Comprehensive comparison of ML classifiers for heart disease prediction.

4. **Mohan, S., Thirumalai, C., & Srivastava, G. (2019).**
   *"Effective Heart Disease Prediction Using Hybrid Machine Learning Techniques."*
   IEEE Access, 7, 81542–81554.
   → Hybrid Random Forest + Linear Model approach; inspired the ensemble direction of this project.

5. **Latha, C. B. C., & Jeeva, S. C. (2019).**
   *"Improving the accuracy of prediction of heart disease risk based on ensemble classification techniques."*
   Informatics in Medicine Unlocked, 16, 100203.
   → Validates ensemble methods (Voting, Bagging, Boosting) for heart disease.

6. **Breiman, L. (2001).**
   *"Random Forests."*
   Machine Learning, 45(1), 5–32.
   → Foundational paper for the **Random Forest** algorithm used in this project.

7. **Friedman, J. H. (2001).**
   *"Greedy Function Approximation: A Gradient Boosting Machine."*
   The Annals of Statistics, 29(5), 1189–1232.
   → Foundational paper for **Gradient Boosting** (basis of HDP-DTRF).

8. **Chen, T., & Guestrin, C. (2016).**
   *"XGBoost: A Scalable Tree Boosting System."*
   Proceedings of the 22nd ACM SIGKDD, 785–794.
   → Paper for **XGBoost** used in model comparison notebook.
