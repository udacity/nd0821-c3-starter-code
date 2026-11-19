from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from starter.ml.data import clean_data, process_data
from starter.ml.model import inference, retrieve_model
from starter.train_model import cat_features

app = FastAPI()


class Record(BaseModel):
    model_config = ConfigDict(
        {"alias_generator": lambda x: x.replace("_", "-")}
    )
    age: int
    workclass: str | None = None
    fnlgt: int
    education: str | None = None
    education_num: int
    marital_status: str | None = None
    occupation: str | None = None
    relationship: str | None = None
    race: str | None = None
    sex: str | None = None
    capital_gain: int = 0
    capital_loss: int = 0
    hours_per_week: int
    native_country: str | None = None
    salary: Literal["<=50K", ">50K"] | None = None


@app.get("/")
def index():
    """Greeting page for FastAPI interface

    Returns:
        message: Simple welcoming string
    """
    return JSONResponse(
        {"message": "Hello and welcome to simple inference API"}
    )


@app.post("/predict")
def make_inference(
    file: Annotated[
        UploadFile, File(description="CSV file with data to predict data for")
    ],
):

    try:
        raw = pd.read_csv(
            file.file, skipinitialspace=True, na_values=["?", ""]
        )  # noqa: E501
        raw = clean_data(raw)
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "Could not parse CSV")

    model, encoder, lb = retrieve_model()
    if not model:
        raise HTTPException(
            500,
            (
                "No model available."
                "Please use `train_model` script to make a new model"
            ),
        )
    df = raw.copy()
    try:
        TypeAdapter(list[Record]).validate_python(df.to_dict("records"))
    except ValidationError as e:
        raise HTTPException(422, e.errors(include_url=False)[:20])

    data, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, data)

    return JSONResponse({"predictions": lb.inverse_transform(preds).tolist()})
