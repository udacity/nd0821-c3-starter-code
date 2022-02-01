from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import *
import pandas as pd
import joblib

# Instantiate the app.
app = FastAPI()

model = joblib.load('model/model.pkl')
encoder = joblib.load('model/encoder.pkl')
lb = joblib.load('model/lb.pkl')


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to the model api"}


class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')
    
    class Config:
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
                "native-country": "United-States"              
            }
        }    
    
   
    
@app.post("/")
async def run_inference(body: Data):   
    data_to_predict_dict = body.dict(by_alias=True)
    data_to_predict = pd.DataFrame(data_to_predict_dict, index=[0])
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
    X, _, _, _ = process_data(
        data_to_predict, 
        categorical_features=cat_features,
        encoder = encoder,
        lb = lb,
        training=False
    )
    inf_value = int(inference(model, X))    
    return {"prediction": inf_value}
