# Put the code for your API here.
from fastapi import FastAPI
from pandas import DataFrame
from src.ml.model import inference, load_model
from src.config import cat_features
from src.ml.data import process_data

from pydantic import BaseModel

app = FastAPI()


class Data(BaseModel):
    age: int = 31
    workclass: str = "Local-gov"
    fnlgt: int = 189265
    education: str = "HS-grad"
    education_num: int = 9
    marital_status: str = "Never-married"
    occupation: str = "Adm-clerical"
    relationship: str = "Not-in-family"
    race: str = "White"
    sex: str = "Female"
    capital_gain: int = 0
    capital_loss: int = 0
    hours_per_week: int = 40
    native_country: str = "United-States"


@app.get("/")
async def greeting():
    response = "Welcome to the small vector machines for Adult income prediction endpoint"
    return response


@app.post("/infer")
async def predict(data: Data):
    model, encoder, lb = load_model()

    input_data = {
            "age":  data.age,
            "workclass": data.workclass,
            "fnlgt": data.fnlgt,
            "education": data.education,
            "education-num": data.education_num,
            "marital-status": data.marital_status,
            "occupation": data.occupation,
            "relationship": data.relationship,
            "race": data.race,
            "sex": data.sex,
            "capital-gain": data.capital_gain,
            "capital-loss": data.capital_loss,
            "hours-per-week": data.hours_per_week,
            "native-country": data.native_country,
        }

    request_df = DataFrame(input_data, index=[0])

    X, _, _, _ = process_data(
        X=request_df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    prediction = inference(model=model, X=X)
    prediction_label = ">50K"
    if prediction == 0:
        prediction_label = "<=50K"

    response = f"Predicted income: {prediction_label}"
    return response
