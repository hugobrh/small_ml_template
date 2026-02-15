from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Credit-G Prediction API")
model = joblib.load("models/lr.joblib")


class CreditInput(BaseModel):
    checking_status: str
    duration: int
    credit_history: str
    purpose: str
    credit_amount: int
    savings_status: str
    employment: str
    installment_commitment: int
    personal_status: str
    other_parties: str
    residence_since: int
    property_magnitude: str
    age: int
    other_payment_plans: str
    housing: str
    existing_credits: int
    job: str
    num_dependents: int
    own_telephone: str
    foreign_worker: str


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: List[CreditInput]):
    df = pd.DataFrame([d.model_dump() for d in data])
    prediction = model.predict(df)
    return {"predictions": prediction.tolist()}
