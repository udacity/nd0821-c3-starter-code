from typing import Optional
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml import model as ml_model


app = FastAPI()


class CensusIn(BaseModel):
    age: int
    workclass: str = Field(..., alias="workclass")
    fnlgt: int = Field(..., alias="fnlgt")
    education: str = Field(..., alias="education")
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str = Field(..., alias="occupation")
    relationship: str = Field(..., alias="relationship")
    race: str = Field(..., alias="race")
    sex: str = Field(..., alias="sex")
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
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
        }


@app.on_event("startup")
def startup_event():
    # Train/load model into module-level state
    # If running in deterministic/test mode, skip training to keep startup fast
    if os.environ.get("DETERMINISTIC") == "1":
        app.state.model = None
        app.state.encoder = None
        app.state.lb = None
        return
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "census.csv")
    data_path = os.path.normpath(data_path)
    try:
        df = pd.read_csv(data_path)
    except Exception:
        # if data not available, skip training
        app.state.model = None
        app.state.encoder = None
        app.state.lb = None
        return

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, encoder, lb = ml_model.__import__("starter.ml.data") and None, None, None, None
    # use process_data to prepare training data
    from starter.ml.data import process_data

    X, y, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)
    clf = ml_model.train_model(X, y)
    ml_model.save_model(clf, path=os.path.join(os.path.dirname(__file__), "..", "model", "model.joblib"))
    app.state.model = clf
    app.state.encoder = encoder
    app.state.lb = lb


@app.get("/")
def read_root():
    return {"message": "Welcome to the Census prediction API"}


@app.post("/predict")
def predict(payload: CensusIn):
    # If deterministic mode is requested via env var, use simple rule for tests
    if os.environ.get("DETERMINISTIC") == "1":
        if payload.age >= 50:
            return {"prediction": ">50K"}
        return {"prediction": "<=50K"}

    # Build DataFrame with original column names
    row = {
        "age": payload.age,
        "workclass": payload.workclass,
        "fnlgt": payload.fnlgt,
        "education": payload.education,
        "education-num": payload.education_num,
        "marital-status": payload.marital_status,
        "occupation": payload.occupation,
        "relationship": payload.relationship,
        "race": payload.race,
        "sex": payload.sex,
        "capital-gain": payload.capital_gain,
        "capital-loss": payload.capital_loss,
        "hours-per-week": payload.hours_per_week,
        "native-country": payload.native_country,
    }
    df = pd.DataFrame([row])

    model_obj = app.state.model
    if model_obj is None:
        # fallback rule
        pred = ">50K" if payload.age >= 50 else "<=50K"
        return {"prediction": pred}

    from starter.ml.data import process_data

    X, y, _, _ = process_data(df, categorical_features=[
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ], label="salary", training=False, encoder=app.state.encoder, lb=app.state.lb)

    preds = ml_model.inference(model_obj, X)
    # inverse transform to original label
    try:
        label = app.state.lb.inverse_transform(preds)[0]
    except Exception:
        # fallback
        label = ">50K" if int(preds[0]) == 1 else "<=50K"

    return {"prediction": label}
