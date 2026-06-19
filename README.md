# Hunger Prediction Project

This is a Flask web application for predicting hunger risk using a trained logistic regression model.

## Project Structure

- `app.py` - Flask application that serves the dashboard and prediction API.
- `train_model.py` - Script to preprocess data, train the model, evaluate it, and save model artifacts.
- `dataset.csv` - Dataset used for model training.
- `hunger_model.pkl` - Trained logistic regression model.
- `scaler.pkl` - Saved `StandardScaler` used for feature scaling.
- `label_encoder.pkl` - Saved `LabelEncoder` used to decode model predictions.
- `templates/` - HTML templates for the dashboard and predictor pages.
- `static/` - Static assets for the web UI.

## Pipeline Architecture

The Hunger Prediction Project follows a modular machine learning pipeline architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  dataset.csv     │  (Raw Data)
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│         train_model.py                   │
│  (Training Pipeline)                     │
│  ├─ Load Dataset                         │
│  ├─ Data Preprocessing                   │
│  │   ├─ Drop unnecessary columns         │
│  │   ├─ Fill missing values              │
│  │   └─ Remove duplicates                │
│  ├─ Label Encoding                       │
│  ├─ Feature Selection                    │
│  ├─ Train-Test Split                     │
│  ├─ Model Training (Logistic Regression)│
│  ├─ Model Evaluation                     │
│  └─ Save Artifacts                       │
└──────────────────────────────────────────┘
         │
    ┌────┴─────────┬──────────────┐
    ▼              ▼              ▼
 hunger_model.pkl  scaler.pkl  label_encoder.pkl
 (Trained Model)   (Scaler)    (Encoder)
    │              │              │
    └──────┬───────┴──────┬───────┘
           │              │
           ▼              ▼
    ┌──────────────┐  ┌──────────────────┐
    │   app.py     │  │ streamlit_app.py │
    │ (Flask API)  │  │  (Streamlit UI)  │
    └──────┬───────┘  └────────┬─────────┘
           │                   │
           ├─────┬─────────────┤
           │     │             │
           ▼     ▼             ▼
      [/predict] [Dashboard] [Interactive]
      [/]        [/predictor] [Visualizations]
      
```

### Pipeline Stages

1. **Data Preparation** (`train_model.py`)
   - Load raw data from `dataset.csv`
   - Remove non-predictive columns (Country, predicted_risk, confidence, abs_change, pct_change)
   - Handle missing values using mean imputation
   - Remove duplicate records

2. **Feature Engineering**
   - Label encode the target variable (`risk_label`)
   - Select features (Rank, 2000, 2008, 2015, 2023)
   - Scale features using StandardScaler

3. **Model Training**
   - Algorithm: Logistic Regression
   - Train-test split for evaluation
   - Performance metrics: Accuracy, Confusion Matrix, Classification Report

4. **Model Serialization**
   - Save trained model to `hunger_model.pkl`
   - Save scaler to `scaler.pkl`
   - Save label encoder to `label_encoder.pkl`

5. **Inference & Deployment**
   - **Flask API** (`app.py`): REST endpoint for predictions
     - `/predict` - POST endpoint accepts JSON with features and returns predictions
   - **Streamlit Dashboard** (`streamlit_app.py`): Interactive web interface with visualizations

## Requirements

- Python 3.14+ (or compatible Python 3.x)
- Flask
- pandas
- scikit-learn
- joblib
- numpy

## Setup

1. Open a terminal in the project folder.
2. Install dependencies:

```powershell
python -m pip install flask pandas scikit-learn joblib numpy
```

## Run the App

Start the Flask server:

```powershell
python app.py
```

Then open the app in your browser at:

```text
http://127.0.0.1:5000
```

## Train the Model

If you want to retrain the model, run:

```powershell
python train_model.py
```

This will generate:

- `hunger_model.pkl`
- `scaler.pkl`
- `label_encoder.pkl`

## Notes

- The app expects the model files (`hunger_model.pkl`, `scaler.pkl`, `label_encoder.pkl`) to exist in the project root.
- The prediction endpoint is available at `/predict` and expects JSON input with the following fields:
  - `rank`
  - `y2000`
  - `y2008`
  - `y2015`
  - `y2023`

Example request body:

```json
{
  "rank": 1,
  "y2000": 10.5,
  "y2008": 12.2,
  "y2015": 14.0,
  "y2023": 15.8
}
```
