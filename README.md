# Diabetes-Prediction-Web-App
👇

🩺 Diabetes Prediction Web App

A machine learning–based web application that predicts whether a person is likely diabetic based on medical inputs.
The model uses a Deep Neural Network (DNN) and is deployed using Streamlit.

🚀 Tech Stack

Python

TensorFlow / Keras

Scikit-Learn

NumPy, Pandas

Streamlit

Joblib

📌 Project Overview

This project predicts diabetes likelihood using a Deep Neural Network trained on a diabetes dataset.
Users enter health-related inputs in the web app, and the model predicts diabetes probability in real-time.

The workflow includes:

Data preprocessing

Scaling

DNN model training

Saving model (.h5) and scaler (.pkl)

Streamlit deployment

🧠 Features

✔ Deep Learning-based Diabetes Prediction
✔ Real-time prediction
✔ Clean Streamlit UI
✔ Input validation
✔ Probability + Risk Prediction
✔ Fully reproducible code
✔ Ready for deployment

📂 Project Structure
diabetes-prediction/
│── app.py
│── model_training.py
│── create_dataset.py
│── requirements.txt
│── data/
│     └── diabetes.csv
│── model/
│     ├── diabetes_model.h5
│     └── scaler.pkl
│── images/
      ├── app_ui.png
      ├── result.png



🔧 How to Run Project Locally
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Generate dataset (optional)
python create_dataset.py

3️⃣ Train the model
python model_training.py

4️⃣ Run Streamlit app
streamlit run app.py

🧪 Model Details
🔹 Model Type:

Deep Neural Network (DNN)

🔹 Layers:

Dense(64, activation='relu')

Dropout(0.3)

Dense(32, activation='relu')

Dropout(0.2)

Dense(1, activation='sigmoid')

🔹 Optimizer:
Adam

🔹 Loss:
Binary Crossentropy

📈 Improvements for Future

Add logistic regression / random forest comparison

Add SHAP for explainable AI


Add authentication system

Create a better UI dashboard

✨ Author

Himanshu Pal
AI/ML Engineer | Python Developer
📧 hpcrc2005@gmail.com

