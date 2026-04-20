from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model + encoders
model = joblib.load("models/best_model.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")
feature_encoders = joblib.load("models/feature_encoders.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

@app.post("/predict")
def predict(features: dict):
    # Convert input to DataFrame
    df = pd.DataFrame([features])

    # Apply encoders safely
    for col, encoder in feature_encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except Exception:
                # Handle unseen categories by defaulting to first known class
                df[col] = encoder.transform([encoder.classes_[0]])

    # Ensure all required columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = None  # fill missing with None or default

    # Reorder columns
    df = df[feature_columns]

    # Predict
    pred = model.predict(df)
    decoded = target_encoder.inverse_transform(pred)

    return {"prediction": decoded[0]}
