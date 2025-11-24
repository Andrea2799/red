# app.py
import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

DATA_PATH = "dataset.csv"
MODEL_REG_PATH = "model_reg.h5"
MODEL_CLF_PATH = "model_clf.h5"
SCALER_PATH = "scaler.pkl"
THRESHOLD = 34.05   # umbral para aprobar/reprobar

def train_and_save_models():
    df = pd.read_csv(DATA_PATH)

    # Usamos las 4 columnas como entrada
    X = df[["hours_studied", "sleep_hours", "attendance_percent", "previous_scores"]].values
    y_reg = df["exam_score"].values
    y_clf = (df["exam_score"] >= THRESHOLD).astype(int).values  # Umbral de aprobación

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    X_train, X_test, y_reg_train, y_reg_test = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)
    _, _, y_clf_train, y_clf_test = train_test_split(X_scaled, y_clf, test_size=0.2, random_state=42)

    # Modelo de regresión
    model_reg = Sequential([
        Dense(32, activation="relu", input_shape=(4,)),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model_reg.compile(optimizer=Adam(learning_rate=0.005), loss="mse", metrics=["mae"])
    model_reg.fit(X_train, y_reg_train, epochs=80, batch_size=8, validation_data=(X_test, y_reg_test), verbose=0)
    model_reg.save(MODEL_REG_PATH)

    # Modelo de clasificación
    model_clf = Sequential([
        Dense(32, activation="relu", input_shape=(4,)),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model_clf.compile(optimizer=Adam(learning_rate=0.005), loss="binary_crossentropy", metrics=["accuracy"])
    model_clf.fit(X_train, y_clf_train, epochs=80, batch_size=8, validation_data=(X_test, y_clf_test), verbose=0)
    model_clf.save(MODEL_CLF_PATH)


# Entrenamos los modelos si no existen
if not (os.path.exists(MODEL_REG_PATH) and os.path.exists(MODEL_CLF_PATH) and os.path.exists(SCALER_PATH)):
    print("[INFO] Modelos no encontrados — entrenando modelos...")
    train_and_save_models()
else:
    print("[INFO] Modelos encontrados. Cargando...")

# Cargamos scaler y modelos
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

model_reg = load_model(MODEL_REG_PATH, compile=False)
model_clf = load_model(MODEL_CLF_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    prob = None
    status = None

    if request.method == "POST":
        try:
            hours = float(request.form.get("hours_studied", 0))
            sleep = float(request.form.get("sleep_hours", 0))
            attendance = float(request.form.get("attendance_percent", 0))
            nota_prev = float(request.form.get("nota_anterior", 0))

            X = np.array([[hours, sleep, attendance, nota_prev]])
            X_scaled = scaler.transform(X)

            exam_pred = float(model_reg.predict(X_scaled, verbose=0)[0][0])
            prob_pred = float(model_clf.predict(X_scaled, verbose=0)[0][0])
            lbl = "Aprobado" if prob_pred >= 0.5 else "Reprobado"

            result = round(exam_pred, 2)
            prob = round(prob_pred, 3)
            status = lbl

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html", predicted_score=result, probability=prob, status=status)


if __name__ == "__main__":
    app.run(debug=True)
