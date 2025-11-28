# model_training.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def load_data():
    df = pd.read_csv("data/diabetes.csv")

    features = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
                "BMI","DiabetesPedigreeFunction","Age"]

    X = df[features].values
    y = df["Outcome"].values

    return X, y

def build_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_model(X_train_scaled.shape[1])

    early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    model.save("model/diabetes_model.h5")
    joblib.dump(scaler, "model/scaler.pkl")

    print("Model Saved → model/diabetes_model.h5")
    print("Scaler Saved → model/scaler.pkl")
