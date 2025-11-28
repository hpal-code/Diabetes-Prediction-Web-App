import streamlit as st
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

# ------------------------------
# Streamlit App Configuration
# ------------------------------
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🩺 Diabetes Prediction Web App")
st.write("Enter the patient's health details below to predict diabetes probability.")

# ------------------------------
# Input Fields
# ------------------------------
Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
Glucose = st.number_input("Glucose (mg/dL)", 50.0, 300.0, 120.0)
BloodPressure = st.number_input("Blood Pressure (mm Hg)", 20.0, 150.0, 70.0)
SkinThickness = st.number_input("Skin Thickness (mm)", 0.0, 99.0, 20.0)
Insulin = st.number_input("Insulin (mu U/ml)", 0.0, 1000.0, 80.0)
BMI = st.number_input("BMI", 10.0, 70.0, 28.0)
DPF = st.number_input("Diabetes Pedigree Function", 0.0, 5.0, 0.5, format="%.3f")
Age = st.number_input("Age", 10, 120, 33)

# ------------------------------
# File Paths
# ------------------------------
MODEL_PATH = "model/diabetes_model.h5"
SCALER_PATH = "model/scaler.pkl"


# ------------------------------
# Load Model & Scaler
# ------------------------------
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        st.error("❌ Model or scaler not found.\n"
                 "👉 Run 'model_training.py' to generate 'diabetes_model.h5' & 'scaler.pkl'.")
        return None, None
    
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"⚠ Error loading model/scaler:\n{e}")
        return None, None


# ------------------------------
# Prediction Button
# ------------------------------
if st.button("🔍 Predict"):
    model, scaler = load_model_and_scaler()
    if model is None:
        st.stop()

    # Prepare input
    X = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                   Insulin, BMI, DPF, Age]])

    X_scaled = scaler.transform(X)

    # Predict Probability
    prob = model.predict(X_scaled)[0][0]

    st.write(f"### 🧪 Diabetes Probability: **`{prob:.4f}`**")

    # Final Classification
    if prob >= 0.5:
        st.error("🔴 **Result: Person is Likely Diabetic (Positive)**")
    else:
        st.success("🟢 **Result: Person is NOT Likely Diabetic (Negative)**")

# ------------------------------
# Footer Message
# ------------------------------
st.markdown("---")
st.info("⚠ This is a demo model trained on sample data. Do not use for medical diagnosis.")
