"""FastAPI app for making prediction on census model"""
import os
import logging
import pandas as pd
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from joblib import load

from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull train_model") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI(title="Marcus Census Prediction 🤖",
              description="Machine Learning trained on the [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income)")
model = load("./starter/model/census_classifier.joblib")
encoder = load("./starter/model/census_encoder.joblib")
lb = load("./starter/model/census_lb.joblib")


class CensusRequest(BaseModel):
    """Census prediction data model."""
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


class CensusResponse(BaseModel):
    """Census prediction response data model."""
    prediction: str


@app.get("/", summary="Root path API endpoint", description="Welcome to the Marcus Census Prediction API!")
async def root():
    return {"message": "Hello World!"}


@app.get("/ping", summary="Ping API endpoint", description="Ping the API to check if it's up.")
async def ping():
    return {"ping": "pong!"}


@app.post("/predict", summary="Prediction API endpoint",
          description="Make a prediction on the census data.",
          response_model=CensusResponse)
async def predict_census(request: CensusRequest = Body(default=None, examples={
    "below": {
        "summary": "<=50K",
        "description": "Below 50K",
        "value": {
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
            "native-country": "United-States"
        }},
    "above": {
        "summary": ">50K",
        "description": "Above 50K",
        "value": {"age": 31, "workclass": "Private", "fnlgt": 45781, "education": "Masters", "education-num": 14,
                  "marital-status": "Never-married", "occupation": "Prof-specialty", "relationship": "Not-in-family",
                  "race": "White", "sex": "Female", "capital-gain": 14084, "capital-loss": 0, "hours-per-week": 50,
                  "native-country": "United-States"}
    }
})):
    input_df = pd.DataFrame.from_dict([request.dict(by_alias=True)])
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']
    X, _, _, _ = process_data(input_df, categorical_features, None, training=False, encoder=encoder, lb=lb)
    pred = inference(model, X)

    return {"prediction": "<=50K" if pred <= 0.5 else ">50K"}
