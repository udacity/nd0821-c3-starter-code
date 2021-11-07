# Put the code for your API here.
import os
import logging
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from joblib import load

from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

app = FastAPI()
logger.info("start %s", os.getcwd())
# starter/main.py
# starter/model/census_classifier.joblib
model = load("./starter/model/census_classifier.joblib")
encoder = load("./starter/model/census_encoder.joblib")
lb = load("./starter/model/census_lb.joblib")


class CensusRequest(BaseModel):
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


@app.get("/")
def root():
    return {"message": "Hello World!"}


@app.get("/ping")
def ping():
    return {"ping": "pong!"}


@app.post("/predict")
def predict_census(request: CensusRequest):
    input_df = pd.DataFrame.from_dict([request.dict(by_alias=True)])
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']
    X, _, _, _ = process_data(input_df, categorical_features, None, training=False, encoder=encoder, lb=lb)
    pred = inference(model, X)

    return {"prediction": "<=50K" if pred <= 0.5 else ">50K"}
