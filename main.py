from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
from starter.starter.ml.model import inference
import pandas as pd
from typing import List
from starter.starter.ml.data import process_data
app = FastAPI()

def to_hyphen(string: str) -> str:
    return string.replace("_", "-")

class PersonInfo(BaseModel):
    age: float
    workclass: str
    fnlgt: float
    education: str
    education_num: float
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: str

    class Config:
        alias_generator = to_hyphen
    
@app.get("/")
async def get_message():
    return {"message": "welcome"}

@app.post("/")
async def post_inference(person_info: PersonInfo):
    # Transform object to pandas data frame
    person_info_dict = person_info.dict(by_alias=True)
    person_info_df = pd.DataFrame([person_info_dict])

    # Load model and encoder, lb
    saved_model = load("starter/model/model.joblib")
    encoder = load("starter/model/encoder.joblib")
    lb = load("starter/model/lb.joblib")

    # Process data using encoder, lb
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
    ps_np, _, _, _ = process_data(
        person_info_df,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        training=False
    )

    # Predict result
    result = inference(saved_model, ps_np)
    return {
        "predict": int(result[0])
    }
