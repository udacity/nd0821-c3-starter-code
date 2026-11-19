from typing import Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from starter.ml.data import clean_data, process_data
from starter.ml.model import inference, retrieve_model
from starter.train_model import cat_features

app = FastAPI()


class Record(BaseModel):
    model_config = ConfigDict(
        {
            "alias_generator": lambda x: x.replace("_", "-"),
        }
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


class PredictResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"predictions": ["<=50K", ">50K"]},
            ]
        }
    )
    predictions: Literal["<=50K", ">50K"] = Field(
        description="One prediction per CSV row, in the same order"
    )


class ErrorResponse(BaseModel):
    detail: str | list[dict]


def error_response(description: str, **examples):
    """Build a responses entry with named examples."""
    return {
        "model": ErrorResponse,
        "description": description,
        "content": {
            "application/json": {
                "examples": {
                    name: {"summary": name.replace("_", " "), "value": value}
                    for name, value in examples.items()
                }
            }
        },
    }


@app.get("/")
def index():
    """Greeting page for FastAPI interface

    Returns:
        message: Simple welcoming string
    """
    return JSONResponse(
        {"message": "Hello and welcome to simple inference API"}
    )


EXAMPLE_CSV = """\
age, workclass, fnlgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country
39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States
50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States
"""  # noqa: E501


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: error_response(
            "CSV could not be parsed",
            unparseable={"detail": "Could not parse CSV"},
        ),
        422: error_response(
            "Missing columns or invalid values",
            missing_columns={"detail": "Missing columns: ['age']"},
            invalid_value={
                "detail": [
                    {
                        "type": "int_parsing",
                        "loc": [0, "age"],
                        "msg": "Input should be a valid integer",
                        "input": "abc",
                    }
                ]
            },
        ),
        500: error_response(
            "No models available",
            missing_model={"detail": "No model available"},
        ),
    },
)
def make_inference(
    file: UploadFile = File(  # noqa: B008
        description=(
            "CSV file with data to predict salary from\n\n"
            f"```csv\n{EXAMPLE_CSV}```\n\n"
            f"Hyphens or underscores in headers are both accepted."
        )
    ),
):

    try:
        raw = pd.read_csv(
            file.file, skipinitialspace=True, na_values=["?", ""]
        )  # noqa: E501
        raw = clean_data(raw)
    except Exception:  # noqa: BLE001
        raise HTTPException(400, {"details": "Could not parse CSV"})

    model, encoder, lb = retrieve_model()
    if not model:
        raise HTTPException(
            500,
            {
                "details": (
                    "No model available."
                    "Please use `train_model` script to make a new model"
                )
            },
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
