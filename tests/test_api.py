from pathlib import Path

import joblib
from fastapi.testclient import TestClient

from src.credit_model.api import app

# Ensure a dummy model exists so tests pass even when `models/` is gitignored.
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)


class _DummyModel:
    def predict(self, X):
        # return a list of zeros with the same length as input
        try:
            length = len(X)
        except Exception:
            length = 1
        return [0] * length


# write a lightweight joblib file if missing
model_path = models_dir / "lr.joblib"
if not model_path.exists():
    joblib.dump(_DummyModel(), model_path)


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_prediction_format():
    # Données fictives respectant la structure du dataset Credit-G
    payload = [
        {
            "checking_status": "<0",
            "duration": 6,
            "credit_history": "critical/other",
            "purpose": "radio/tv",
            "credit_amount": 1169.0,
            "savings_status": "unknown",
            "employment": "unemployed",
            "installment_commitment": 4,
            "personal_status": "male single",
            "other_parties": "none",
            "residence_since": 4,
            "property_magnitude": "real estate",
            "age": 67,
            "other_payment_plans": "none",
            "housing": "own",
            "existing_credits": 2,
            "job": "skilled",
            "num_dependents": 1,
            "own_telephone": "yes",
            "foreign_worker": "yes",
        }
    ]

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert isinstance(response.json()["predictions"], list)
