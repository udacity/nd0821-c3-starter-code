# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import Literal

from starter.ml.data import process_data
from starter.ml.model import inference

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Income Prediction API!"}

# Define Pydantic model for request body
class InferenceInput(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "examples": [
                {
                    "age": 37,
                    "workclass": "Self-emp-not-inc",
                    "fnlgt": 284582,
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Never-married",
                    "occupation": "Exec-managerial",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 0,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States"
                }
            ]
        }


# Load model and encoders
model = joblib.load("starter/model/model.pkl")
encoder = joblib.load("starter/model/encoder.pkl")
lb = joblib.load("starter/model/label_binarizer.pkl")

@app.post("/inference")
def predict(input_data: InferenceInput):
    print("RECEIVED INPUT:", input_data.dict(by_alias=True))
    input_dict = input_data.dict(by_alias=True)
    data_df = pd.DataFrame([input_dict])

    X, _, _, _ = process_data(
        data_df,
        categorical_features=[
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"
        ],
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )

    pred = inference(model, X)
    prediction_label = lb.inverse_transform(pred)[0]

    return {"prediction": prediction_label}
