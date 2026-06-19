
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
import numpy as np
import joblib


# LOAD DATASET

df = pd.read_csv("dataset.csv")

print("Dataset Loaded Successfully")

# DATA PREPROCESSING


# Remove unnecessary columns
df = df.drop(columns=['Country', 'predicted_risk', 'confidence', 'abs_change', 'pct_change'])

# Handle Missing Values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Remove Duplicate Records
df.drop_duplicates(inplace=True)

print("Data Preprocessing Completed")


# LABEL ENCODING


le = LabelEncoder()

df['risk_label'] = le.fit_transform(df['risk_label'])

print("Label Encoding Completed")

# FEATURE SELECTION

X = df.drop('risk_label', axis=1)
y = df['risk_label']

print("Feature Selection Completed")


# FEATURE SCALING


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Feature Scaling Completed")


# TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print("Train Test Split Completed")


# LOGISTIC REGRESSION MODEL


model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

print("Model Training Completed")

# PREDICTION

y_pred = model.predict(X_test)
# MODEL EVALUATION


accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# No regression evaluation metrics (classification task)


# SAVE MODEL FILES


joblib.dump(model, "hunger_model.pkl")

joblib.dump(scaler, "scaler.pkl")

joblib.dump(le, "label_encoder.pkl")

print("\nAll Files Saved Successfully")
