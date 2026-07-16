from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="API de Scoring Crédit")

# Charger le modèle, le scaler et les noms de features au démarrage
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# Schéma de validation des données d'entrée (Pydantic)
class CreditData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float
    BILL_TO_LIMIT_RATIO: float
    AVG_PAY_AMT: float
    MONTHS_LATE: int


@app.get("/")
def root():
    return {"message": "API de scoring crédit opérationnelle"}


@app.post("/predict")
def predict(data: CreditData):
    try:
        # Convertir en DataFrame dans le bon ordre de colonnes
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])[feature_names]

        # Scaling (même transformation que l'entraînement)
        X_scaled = scaler.transform(df)

        # Prédiction
        proba = model.predict_proba(X_scaled)[0, 1]
        prediction = int(proba >= 0.5)

        return {
            "prediction": prediction,
            "probability_default": round(float(proba), 4),
            "risk_level": "high" if proba >= 0.5 else "low"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))