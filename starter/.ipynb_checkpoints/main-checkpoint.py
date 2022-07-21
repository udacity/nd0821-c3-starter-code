# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union 
from starter.ml.data import process_data
from starter.ml.model import *
from starter import train_model as tm

from fastapi import FastAPI, Header
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel


# Initialize an instance of FastAPI
app = FastAPI()


class predictset(BaseModel):
    data_path : str
    model_path : str

class sliceset(BaseModel):
    data_path : str
    model_path : str
    feature : str

@app.get("/")
def root():
    return {"message": "Welcome to Your Sentiment Classification FastAPI"}

@app.post("/predict")
def predict(settings: predictset):
    precision, recall, fbeta = tm.get_model(settings.data_path, settings.model_path)
    return {"precision": precision,
           "recall": recall,
           "f1 score": fbeta}
    
@app.post("/slice_test")
def slice_test(settings: sliceset):
    tm.splice_testing(settings.data_path, settings.model_path, settings.feature)
    return {"feature slice tested": settings.feature}