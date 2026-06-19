
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("hunger_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    return render_template("dashboard.html")

@app.route('/predictor')
def predictor():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    data = request.json

    features = pd.DataFrame([{
        "Rank": float(data["rank"]),
        "2000": float(data["y2000"]),
        "2008": float(data["y2008"]),
        "2015": float(data["y2015"]),
        "2023": float(data["y2023"])
    }])

    scaled = scaler.transform(features)

    prediction = model.predict(scaled)

    result = label_encoder.inverse_transform(prediction)[0]

    return jsonify({
        "prediction": result
    })

if __name__ == "__main__":
    app.run(debug=True)

