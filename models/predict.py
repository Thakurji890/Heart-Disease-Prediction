"""
predict.py — run the trained heart-disease model on new patient data.

Usage:
    python predict.py

Requires: models/preprocessor.pkl, models/iqr_bounds.pkl, models/random_forest.pkl
(all three must be present, in that models/ folder relative to this script,
or edit the paths below).
"""

import pandas as pd
import joblib

# ---- 1. Load the saved pipeline pieces ----
preprocessor = joblib.load("models/preprocessor.pkl")
iqr_bounds = joblib.load("models/iqr_bounds.pkl")
model = joblib.load("models/hdp_dtrf.pkl")

NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def predict(patients: pd.DataFrame) -> pd.DataFrame:
    """
    patients: a DataFrame with one row per person and these 13 raw columns:
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal

    Returns the same rows with two extra columns: `prediction` (0/1) and
    `probability` (model confidence that target == 1, i.e. heart disease present).
    """
    df = patients.copy()

    
    for col in NUMERIC_COLS:
        lower, upper = iqr_bounds[col]
        df[col] = df[col].clip(lower, upper)

    X = preprocessor.transform(df)
    CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    cat_feature_names = preprocessor.named_transformers_["cat"]["ohe"].get_feature_names_out(CATEGORICAL_COLS)
    X = pd.DataFrame(X, columns=NUMERIC_COLS + list(cat_feature_names)) 

    prediction = model.predict(X)
    probability = model.predict_proba(X)[:, 1]

    result = patients.copy()
    result["prediction"] = prediction
    result["probability_of_disease"] = probability.round(3)
    return result


if __name__ == "__main__":
    # ---- Example: checking ONE person ----
    one_person = pd.DataFrame([{
        "age": 58, "sex": 1, "cp": 0, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }])
    print("Single person result:")
    print(predict(one_person))

    # ---- Example: checking MULTIPLE people at once (a batch/group) ----
    many_people = pd.DataFrame([
        {"age": 58, "sex": 1, "cp": 0, "trestbps": 145, "chol": 233, "fbs": 1,
         "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1},
        {"age": 45, "sex": 0, "cp": 2, "trestbps": 130, "chol": 210, "fbs": 0,
         "restecg": 1, "thalach": 172, "exang": 0, "oldpeak": 0.5, "slope": 2, "ca": 0, "thal": 2},
        {"age": 67, "sex": 1, "cp": 3, "trestbps": 160, "chol": 286, "fbs": 0,
         "restecg": 0, "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 1, "ca": 3, "thal": 2},
    ])
    print("\nMultiple people result:")
    print(predict(many_people))

    # ---- Or load a whole CSV of people and predict for all of them ----
    # many_people = pd.read_csv("data/raw/bulk_patients.csv")
    # results = predict(many_people)
    # results.to_csv("predictions.csv", index=False)
