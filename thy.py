import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# ---------------- TRAIN MODEL ---------------- #

if not os.path.exists("model.pkl"):

    df = pd.read_csv("thyroidDF.csv")

    # Target: Normal = 0, Disease = 1
    df["target"] = (df["target"] != "-").astype(int)

    # Drop ID column
    df.drop(columns=["patient_id"], inplace=True)

    # Features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Convert all text columns into numbers
    X = pd.get_dummies(X, dummy_na=True)

    # Fill missing numeric values
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    joblib.dump(model, "model.pkl")
    joblib.dump(imputer, "imputer.pkl")
    joblib.dump(list(pd.get_dummies(df.drop("target", axis=1), dummy_na=True).columns), "columns.pkl")

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("model.pkl")
imputer = joblib.load("imputer.pkl")
columns = joblib.load("columns.pkl")

# ---------------- STREAMLIT ---------------- #

st.title("🩺 Thyroid Disease Prediction")

age = st.number_input("Age", 1, 100, 30)

sex = st.selectbox("Sex", ["F", "M"])

TSH = st.number_input("TSH", 0.0)
T3 = st.number_input("T3", 0.0)
TT4 = st.number_input("TT4", 0.0)
T4U = st.number_input("T4U", 0.0)
FTI = st.number_input("FTI", 0.0)

if st.button("Predict"):

    row = {c: 0 for c in columns}

    if "age" in row:
        row["age"] = age
    if "TSH" in row:
        row["TSH"] = TSH
    if "T3" in row:
        row["T3"] = T3
    if "TT4" in row:
        row["TT4"] = TT4
    if "T4U" in row:
        row["T4U"] = T4U
    if "FTI" in row:
        row["FTI"] = FTI

    # measured flags
    for c in ["TSH_measured_t", "T3_measured_t", "TT4_measured_t", "T4U_measured_t", "FTI_measured_t"]:
        if c in row:
            row[c] = 1

    # sex
    if sex == "F" and "sex_F" in row:
        row["sex_F"] = 1
    if sex == "M" and "sex_M" in row:
        row["sex_M"] = 1

    sample = pd.DataFrame([row], columns=columns)

    sample = imputer.transform(sample)

    pred = model.predict(sample)

    if pred[0] == 1:
        st.error("⚠ Thyroid Disease Detected")
    else:
        st.success("✅ Normal")