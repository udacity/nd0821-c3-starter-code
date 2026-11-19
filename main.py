import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, TypeAdapter, ValidationError

from starter.ml.data import clean_data, process_data
from starter.ml.model import inference, retrieve_model
from starter.train_model import cat_features

app = FastAPI()


class Record(BaseModel):
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
def make_inference(file: UploadFile):
    try:
        df = pd.read_csv(file.file, skipinitialspace=True, na_values=["?", ""])
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "Could not parse CSV")

    df = clean_data(df)

    model, encoder, lb = retrieve_model()
    if not model:
        raise HTTPException(
            500,
            (
                "No model available."
                "Please use `train_model` script to make a new model"
            ),
        )
    missing = set(Record.model_fields) - set(
        df.columns.str.strip().str.replace("-", "_").str.lower()
    )
    if missing:
        raise HTTPException(422, f"Missing columns: {sorted(missing)}")

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
