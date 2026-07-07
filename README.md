# Heart Disease Prediction

This repository contains an end-to-end machine learning project for predicting heart disease. The project explores a dataset containing various medical indicators to build robust classification models that predict the presence or absence of heart disease in patients.

## Project Structure

The project is broken down into structured Jupyter notebooks that document the workflow from initial analysis to final model selection:

1. **`01_EDA.ipynb`** - **Exploratory Data Analysis**
   - Data loading and initial inspection
   - Statistical summaries
   - Univariate and bivariate analysis (Histograms, Boxplots)
   - Correlation heatmap to explore relationships between features

2. **`02_Preprocessing.ipynb`** - **Data Preprocessing & Feature Engineering**
   - Handling missing values (Imputation)
   - Removing duplicate records
   - Outlier treatment using the IQR method
   - Feature scaling using `StandardScaler`
   - Train-test splitting (80:20 split with stratification)
   - Feature importance analysis

3. **`03_Baseline_Models.ipynb`** - **Baseline Model Training**
   - Training multiple baseline classifiers: Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, K-Nearest Neighbors, and Naive Bayes.
   - Initial evaluation of models based on Accuracy, Precision, Recall, F1-score, and ROC-AUC.

4. **`04_All_Models_Comparison.ipynb`** - **Model Comparison and Tuning**
   - Incorporating advanced models (like Gradient Boosting and XGBoost)
   - Hyperparameter tuning using `GridSearchCV`
   - Comprehensive model comparison and final selection

## Requirements

To run this project locally, you will need Python 3 installed. The project relies on several data science and machine learning libraries.

You can install all dependencies by running:
```bash
pip install -r requirement.txt
```

### Dependencies Included:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `joblib`

## How to Run
1. Clone this repository to your local machine.
2. Install the required dependencies using the command above.
3. Open Jupyter Notebook or your preferred IDE to explore and run the notebooks in order.
