"""FastAPI application serving the trained census income classifier."""
from pathlib import Path
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from starter.ml.data import process_data
from starter.train_model import CAT_FEATURES, MODEL_DIR
from starter.ml.model import inference


class CensusRecord(BaseModel):
    """One census record; aliases preserve the original CSV column names."""

    model_config = ConfigDict(populate_by_name=True, json_schema_extra={
        "example": {
            "age": 39, "workclass": "State-gov", "fnlwgt": 77516,
            "education": "Bachelors", "education-num": 13,
            "marital-status": "Never-married", "occupation": "Adm-clerical",
            "relationship": "Not-in-family", "race": "White", "sex": "Male",
            "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40,
            "native-country": "United-States",
        }
    })
    age: int = Field(example=39)
    workclass: str = Field(example="State-gov")
    fnlwgt: int = Field(example=77516)
    education: str = Field(example="Bachelors")
    education_num: int = Field(alias="education-num", example=13)
    marital_status: str = Field(alias="marital-status", example="Never-married")
    occupation: str = Field(example="Adm-clerical")
    relationship: str = Field(example="Not-in-family")
    race: str = Field(example="White")
    sex: str = Field(example="Male")
    capital_gain: int = Field(alias="capital-gain", example=2174)
    capital_loss: int = Field(alias="capital-loss", example=0)
    hours_per_week: int = Field(alias="hours-per-week", example=40)
    native_country: str = Field(alias="native-country", example="United-States")


app = FastAPI(title="Census Income Classifier", version="1.0.0")
ROOT = Path(__file__).resolve().parent


def load_artifacts():
    """Load the committed model and preprocessing artifacts."""
    artifacts = []
    for name in ("model.pkl", "encoder.pkl", "label_binarizer.pkl"):
        with (MODEL_DIR / name).open("rb") as file:
            artifacts.append(pickle.load(file))
    return artifacts


@app.get("/")
def read_root() -> dict[str, str]:
    """Return a welcome message."""
    return {"message": "Welcome to the Census Income Classifier API."}


@app.post("/predict")
def predict(record: CensusRecord) -> dict[str, str]:
    """Predict whether income is above $50K for one census record."""
    model, encoder, label_binarizer = load_artifacts()
    row = pd.DataFrame([record.model_dump(by_alias=True)])
    features, _, _, _ = process_data(
        row, categorical_features=CAT_FEATURES, training=False,
        encoder=encoder, lb=label_binarizer,
    )
    prediction = inference(model, features)[0]
    return {"prediction": str(label_binarizer.inverse_transform([prediction])[0])}
