from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
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
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str | None = None
    occupation: str
    relationship: str | None = None
    race: str
    sex: str
    capital_gain: int = Field(0)
    capital_loss: int = Field(0)
    hours_per_week: int
    native_country: str | None = None
    salary: str | None = Field("None")


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


class PredictRequest(BaseModel):
    csv_data: list[Record]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "csv_data": [
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
                        "sex": "male",
                        "capital-gain": 2174,
                        "capital-loss": 0,
                        "hours-per-week": 40,
                        "native-country": "United-States",
                        "salary": "<=50K",
                    },
                    {
                        "age": 50,
                        "workclass": "Federal-gov",
                        "fnlgt": 251585,
                        "education": "Bachelors",
                        "education-num": 13,
                        "marital-status": "Divorced",
                        "occupation": "Exec-managerial",
                        "relationship": "Not-in-family",
                        "race": "White",
                        "sex": "male",
                        "capital-gain": 0,
                        "capital-loss": 0,
                        "hours-per-week": 55,
                        "native-country": "United-States",
                        "salary": "<=50K",
                    },
                ]
            }
        }
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


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: error_response(
            "CSV could not be parsed",
            unparseable={"detail": "Could not parse data"},
        ),
        422: error_response(
            "Missing columns or invalid values",
            missing_columns={"detail": "Missing columns: ['age']"},
            invalid_value={
                "detail": [
                    {
                        "type": "missing",
                        "loc": ["body", "csv_data", 0, "age"],
                        "msg": "Field required",
                        "input": {
                            "capital-gain": 2174,
                            "capital-loss": 0,
                            "education": "Bachelors",
                            "education-num": 13,
                            "fnlgt": 77516,
                            "hours-per-week": 40,
                            "marital-status": "Never-married",
                            "native-country": "United-States",
                            "occupation": "Adm-clerical",
                            "race": "White",
                            "relationship": "Not-in-family",
                            "sex": "male",
                        },
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
    request: PredictRequest,
):

    try:
        df = pd.DataFrame(
            [r.model_dump(by_alias=True) for r in request.csv_data]
        )
        df = clean_data(df, missing="unknown")
    except KeyError as e:
        raise HTTPException(
            422,
            {
                "details": [
                    {
                        "type": "missing",
                        "loc": [0, str(e)],
                        "msg": "Input should be a valid integer",
                        "input": [r.model_dump() for r in request.csv_data],
                    }
                ]
            },
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, {"details": str(e)})

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

    try:
        TypeAdapter(list[Record]).validate_python(df.to_dict("records"))
    except ValidationError as e:
        raise HTTPException(422, e.errors(include_url=False)[:20])

    print(df)

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
