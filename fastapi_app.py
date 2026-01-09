# ============================================================
# FastAPI — Titanic Survival Prediction API
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Load Model & Scaler
# ------------------------------------------------------------

MODEL_PATH = "titanic_best_model.joblib"

try:
    model = joblib.load(MODEL_PATH)
except:
    raise RuntimeError("❌ Could not load model. Make sure titanic_best_model.joblib exists.")


# ------------------------------------------------------------
# FastAPI App Settings
# ------------------------------------------------------------

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predict survival probability using trained ML model.",
    version="1.0"
)

# Allow frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# Input Schema for API
# (same features you used to train the model)
# ------------------------------------------------------------

class Passenger(BaseModel):
    Pclass: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Sex_male: int
    Embarked_Q: int
    Embarked_S: int


# ------------------------------------------------------------
# Root Endpoint
# ------------------------------------------------------------

@app.get("/")
def home():
    return {"message": "Titanic Prediction API is running!"}


# ------------------------------------------------------------
# Prediction Endpoint
# ------------------------------------------------------------

@app.post("/predict")
def predict_survival(data: Passenger):

    # Convert input to DataFrame (same order as model training)
    input_df = pd.DataFrame([{
        "Pclass": data.Pclass,
        "Age": data.Age,
        "SibSp": data.SibSp,
        "Parch": data.Parch,
        "Fare": data.Fare,
        "Sex_male": data.Sex_male,
        "Embarked_Q": data.Embarked_Q,
        "Embarked_S": data.Embarked_S,
        "FamilySize": data.SibSp + data.Parch + 1,
        "IsAlone": 1 if (data.SibSp + data.Parch + 1) == 1 else 0,
    }])

    try:
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "survived": int(pred),
        "survival_probability": float(proba)
    }
