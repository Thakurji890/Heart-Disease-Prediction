"""
predict.py — run the trained heart-disease model on new patient data.

Usage:
    python predict.py

Model: XGBoost (Tuned) — Best ROC-AUC model (93.41% AUC, 91.18% Recall)
Requires: models/preprocessor.pkl, models/iqr_bounds.pkl, models/xgboost_best.pkl
(all three must be present, in that models/ folder relative to this script,
or edit the paths below).

IMPORTANT — category codes:
  cp (chest pain type):   1=typical angina, 2=atypical angina, 3=non-anginal pain, 4=asymptomatic
  slope:                  1=upsloping, 2=flat, 3=downsloping
  thal:                   3=normal, 6=fixed defect, 7=reversible defect
  sex, fbs, exang, restecg, ca: same 0/1 (or 0-3 for ca) coding as before.

These match the raw codes actually present in data/raw/heart2.csv, which is what the
preprocessor's OneHotEncoder was fit on. Passing any other codes (e.g. cp=0, thal=1,
slope=0) will NOT raise an obvious error by default — OneHotEncoder silently zeroes
out unrecognized categories — so this script explicitly validates input codes first
and raises a clear error instead of silently producing a wrong prediction.
"""

import pandas as pd
import joblib

# ---- 1. Load the saved pipeline pieces ----
# xgboost_best.pkl: Best ROC-AUC (93.41%), Best Recall (91.18%) — ideal for medical use
preprocessor = joblib.load("models/preprocessor.pkl")
iqr_bounds = joblib.load("models/iqr_bounds.pkl")
model = joblib.load("models/xgboost_best.pkl")

NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# The exact category values the OneHotEncoder was fit on (from data/raw/heart2.csv)
VALID_CODES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}


def _validate_categories(df: pd.DataFrame) -> None:
    """Fail loudly instead of silently, if any category code wasn't seen during training."""
    problems = []
    for col, valid in VALID_CODES.items():
        seen = set(df[col].dropna().unique())
        bad = seen - valid
        if bad:
            problems.append(f"  - '{col}': got {sorted(bad)}, expected one of {sorted(valid)}")
    if problems:
        raise ValueError(
            "Invalid category codes found (these would silently be zeroed out by the "
            "encoder and produce a wrong prediction, so this is being rejected instead):\n"
            + "\n".join(problems)
        )


def predict(patients: pd.DataFrame) -> pd.DataFrame:
    """
    patients: a DataFrame with one row per person and these 13 raw columns:
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    (see the category code table in the module docstring above)

    Returns the same rows with two extra columns: `prediction` (0/1) and
    `probability_of_disease` (model confidence that target == 1).
    """
    df = patients.copy()
    _validate_categories(df)

    # Apply the SAME outlier clipping bounds learned from training data
    for col in NUMERIC_COLS:
        lower, upper = iqr_bounds[col]
        df[col] = df[col].clip(lower, upper)

    # Impute + one-hot encode + scale using the fitted preprocessor (never re-fit on new data!)
    X = preprocessor.transform(df)
    cat_feature_names = preprocessor.named_transformers_["cat"]["ohe"].get_feature_names_out(CATEGORICAL_COLS)
    X = pd.DataFrame(X, columns=NUMERIC_COLS + list(cat_feature_names))

    prediction = model.predict(X)
    probability = model.predict_proba(X)[:, 1]

    result = patients.copy()
    result["prediction"] = prediction
    result["probability_of_disease"] = probability.round(3)
    return result


if __name__ == "__main__":
    # ---- Example: checking ONE person (using CORRECT category codes) ----
    one_person = pd.DataFrame([{
        "age": 58, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 7
    }])
    print("Single person result:")
    print(predict(one_person))

    # ---- Example: checking MULTIPLE people at once (a batch/group) ----
    many_people = pd.DataFrame([
        {"age": 58, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, "fbs": 1,
         "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 7},
        {"age": 45, "sex": 0, "cp": 3, "trestbps": 130, "chol": 210, "fbs": 0,
         "restecg": 1, "thalach": 172, "exang": 0, "oldpeak": 0.5, "slope": 1, "ca": 0, "thal": 3},
        {"age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0,
         "restecg": 0, "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 6},
    ])
    print("\nMultiple people result:")
    print(predict(many_people))

    # ---- Or load a whole CSV of people and predict for all of them ----
    # many_people = pd.read_csv("data/raw/bulk_patients.csv")
    # results = predict(many_people)
    # results.to_csv("predictions.csv", index=False)
