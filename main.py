# Put the code for your API here.
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from starter.ml.data import process_data
from starter.ml.model import inference, load_artifact
from starter.train_model import cat_features

MODEL_DIR = Path(__file__).resolve().parent / "model"

model = load_artifact(MODEL_DIR / "model.joblib")
encoder = load_artifact(MODEL_DIR / "encoder.joblib")
lb = load_artifact(MODEL_DIR / "lb.joblib")


class CensusData(BaseModel):
    """ One person's census record. Hyphenated CSV column names are used as aliases. """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "fnlgt": 77516,
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 2174,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States",
                }
            ]
        },
    )

    # Fields are in the same order as the CSV columns, which process_data relies on.
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


class Prediction(BaseModel):
    prediction: str = Field(examples=["<=50K"])


app = FastAPI(
    title="Census Income Prediction API",
    description="Predicts whether a person earns more than $50K a year from census data.",
    version="1.0.0",
)


@app.get("/")
async def welcome() -> dict[str, str]:
    return {"message": "Welcome to the Census Income Prediction API!"}


@app.post("/predict")
async def predict(data: CensusData) -> Prediction:
    df = pd.DataFrame([data.model_dump(by_alias=True)])
    X, _, _, _ = process_data(
        df, categorical_features=cat_features, training=False, encoder=encoder, lb=lb
    )
    pred = inference(model, X)
    return Prediction(prediction=lb.inverse_transform(pred)[0])
