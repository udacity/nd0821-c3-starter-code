import logging
import os
import pandas as pd
from joblib import load

from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml.data import process_data
from starter.ml.model import inference

logging.basicConfig(level=logging.DEBUG, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()
logger.info("Using dvc")
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

logger.info("Loading models")

encoder = load("model/encoder.joblib")
lb = load("model/lb.joblib")
model = load("model/cl_model.joblib")

app = FastAPI(
    title="API for salary predictor",
    description="This API helps to classify",
    version="0.0.1",
)


class InputData(BaseModel):
    age: int = Field(..., example=36)
    workclass: str = Field(..., example="Federal-gov")
    fnlgt: int = Field(..., example=98350)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=10)
    marital_status: str = Field(
        ..., alias="marital-status", example="Married-civ-spouse"
    )
    occupation: str = Field(..., example="Exec-managerial")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=0)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=60)
    native_country: str = Field(..., alias="native-country", example="Mexico")


@app.get("/")
async def welcome():
    return {"message": "Welcome to the salary predictor"}


@app.post("/predictions")
async def prediction(input_data: InputData):
    logger.info("Input data")
    df = pd.DataFrame.from_dict([input_data.dict(by_alias=True)])
    X, _, _, _ = process_data(
        X=df,
        label=None,
        training=False,
        categorical_features=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ],
        encoder=encoder,
        lb=lb,
    )
    logger.info("Preprocessing")
    pred = inference(model, X)
    logger.info("Inference")
    y = lb.inverse_transform(pred)[0]

    return {"predicted salary": y.strip()}
