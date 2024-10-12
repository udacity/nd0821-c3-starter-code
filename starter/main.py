
#To start the server use the cmd to type: uvicorn main:app --reload

# Put the code for your API here.
import os
import numpy as np
import pandas as pd
#Import libraries related to fastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
#Import the inference function to be used to predict the values
from starter.starter.ml.model import inference
from starter.starter.ml.data import process_data
#Import the model to be used to predict
model = pd.read_pickle(r"starter/model/model.pkl")
Encoder = pd.read_pickle(r"starter/model/encoder.pkl")

#Initial a FastAPI instance
app = FastAPI()

#Give Heroku the ability to pull in data from DVC upon app start up.
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# pydantic models
class DataIn(BaseModel):
    #The input should be alist of 108 values 
    age : int = 39
    workclass : str =  "State-gov"
    fnlgt : int = 77516
    education : str = "Bachelors"
    education_num : int = 13
    marital_status : str = "Never-married"
    occupation : str = "Adm-clerical"
    relationship : str = "Not-in-family"
    race : str = "White"
    sex : str = "Male"
    capital_gain : int = 2174
    capital_loss : int = 0
    hours_per_week : int = 40
    native_country : str = "United-States"

class DataOut(BaseModel):
    #The forecast output will be either >50K or <50K 
    forecast: str = "Income > 50k"

#Adding a Welcome message to the initial page
@app.get("/")
async def root():
    return {"Welcome": "to the Model!"}

# routes
@app.get("/welcome")
async def welcome():
    return {"Welcome": "to the Model!"}


@app.post("/predict", response_model=DataOut, status_code=200)
def get_prediction(payload: DataIn):
    #Reading the input data
    age = payload.age
    workclass = payload.workclass
    fnlgt = payload.fnlgt
    education = payload.education
    education_num = payload.education_num
    marital_status = payload.marital_status
    occupation = payload.occupation
    relationship = payload.relationship
    race = payload.race
    sex = payload.sex
    capital_gain = payload.capital_gain
    capital_loss = payload.capital_loss
    hours_per_week = payload.hours_per_week
    native_country = payload.native_country
    #Converted the inputs into Dataframe to be processed 
    df = pd.DataFrame([{"age" : age,
                        "workclass" : workclass,
                        "fnlgt" : fnlgt,
                        "education" : education,
                        "education-num" : education_num,
                        "marital-status" : marital_status,
                        "occupation" : occupation,
                        "relationship" : relationship,
                        "race" : race,
                        "sex" : sex,
                        "capital-gain" : capital_gain,
                        "capital-loss" : capital_loss,
                        "hours-per-week" : hours_per_week,
                        "native-country" : native_country}])
    # Process the data with the process_data function.
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
    X_processed, y_processed, encoder, lb = process_data(df, categorical_features=cat_features, training=False,encoder=Encoder)
    #Calling the inference function to make a prediction  
    prediction_outcome = inference(model, X_processed)
    
    #Interpreting the prediction for the end user
    if prediction_outcome == 0:
        prediction_outcome = "Income < 50k"
    elif prediction_outcome == 1:
        prediction_outcome = "Income > 50k"
    #Building the response dictionary
    response_object = {"forecast": prediction_outcome}
    return response_object
# Put the code for your API here.
