# create_dataset.py
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import os

# FIX: Create folder if missing
os.makedirs("data", exist_ok=True)

def generate_synthetic_pima(n_samples=1000, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        n_clusters_per_class=2,
        weights=[0.6, 0.4],
        random_state=random_state
    )

    cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
            "BMI","DiabetesPedigreeFunction","Age"]

    df = pd.DataFrame(X, columns=cols)

    # realistic ranges
    df["Pregnancies"] = (np.abs(df["Pregnancies"]) * 2).round().astype(int)
    df["Glucose"] = np.interp(df["Glucose"], (df["Glucose"].min(), df["Glucose"].max()), (70, 200)).round(1)
    df["BloodPressure"] = np.interp(df["BloodPressure"], (df["BloodPressure"].min(), df["BloodPressure"].max()), (40, 120)).round(1)
    df["SkinThickness"] = np.interp(df["SkinThickness"], (df["SkinThickness"].min(), df["SkinThickness"].max()), (5, 45)).round(1)
    df["Insulin"] = np.interp(df["Insulin"], (df["Insulin"].min(), df["Insulin"].max()), (15, 276)).round(1)
    df["BMI"] = np.interp(df["BMI"], (df["BMI"].min(), df["BMI"].max()), (18, 50)).round(1)
    df["DiabetesPedigreeFunction"] = np.abs(df["DiabetesPedigreeFunction"]).round(3)
    df["Age"] = np.interp(df["Age"], (df["Age"].min(), df["Age"].max()), (21, 80)).round().astype(int)

    df["Outcome"] = y
    return df


if __name__ == "__main__":
    df = generate_synthetic_pima(1000)
    df.to_csv("data/diabetes.csv", index=False)
    print("Dataset saved → data/diabetes.csv")
