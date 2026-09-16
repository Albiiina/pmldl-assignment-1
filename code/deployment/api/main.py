from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
MODEL_PATH = Path("/app/models/model.joblib")
if not MODEL_PATH.exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

model: Any = None


class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model when the API starts."""
    global model
    model = joblib.load(MODEL_PATH)
    yield
    model = None


app = FastAPI(title="Titanic Survival Prediction API", lifespan=lifespan)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Titanic Survival Prediction API"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(passenger: Passenger) -> dict[str, int | bool | float]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        passenger_data = passenger.model_dump()
        input_data = pd.DataFrame(
            [[passenger_data[feature] for feature in FEATURES]],
            columns=FEATURES,
        )

        prediction = int(model.predict(input_data)[0])
        result: dict[str, int | bool | float] = {
            "prediction": prediction,
            "survived": prediction == 1,
        }

        if hasattr(model, "predict_proba"):
            class_labels = list(model.classes_)
            survived_class_index = class_labels.index(1)
            probability = model.predict_proba(input_data)[0][survived_class_index]
            result["survival_probability"] = float(probability)

        return result
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error
