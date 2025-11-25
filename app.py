from flask import Flask, render_template, request
import numpy as np
import pickle
import tensorflow as tf

app = Flask(__name__)

# Cargar scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Cargar modelos (aunque no usemos la probabilidad del modelo_clf)
model_reg = tf.keras.models.load_model("model_reg.h5")
model_clf = tf.keras.models.load_model("model_clf.h5")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def predict():
    try:
        # Obtener datos del formulario
        horas_estudio = float(request.form["horas_estudio"])
        horas_suenio = float(request.form["horas_suenio"])
        asistencia = float(request.form["asistencia"])
        nota_anterior = float(request.form["nota_anterior"])

        # Arreglo con los datos en el orden correcto
        X = np.array([[horas_estudio, horas_suenio, asistencia, nota_anterior]])

        # Escalar entradas
        X_scaled = scaler.transform(X)

        # Predicción de nota
        nota_estimada = float(model_reg.predict(X_scaled)[0][0])

        # 🔥 REGLA FIJA PARA APROBAR — LO QUE TÚ PEDISTE
        estado = "Aprobado" if nota_estimada >= 3.0 else "Reprobado"
        prob_aprobar = 1.0 if nota_estimada >= 3.0 else 0.0

        return render_template(
            "index.html",
            nota=round(nota_estimada, 2),
            prob=round(prob_aprobar, 2),
            estado=estado
        )

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
