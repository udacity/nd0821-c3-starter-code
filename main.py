# Put the code for your API here.
import os
from pandas import DataFrame
from fastapi import FastAPI
from api.schema import ModelInput
from starter.ml.data import process_data
from starter.ml.model import load_model, load_encoder, load_lb, inference

# load model
api_model = load_model(os.path.join('model', 'model_dtc.pkl'))

# Categorical Features
cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {'greeting': 'Hello World!'}

@app.post("/predict")
async def predict(input_data: ModelInput):
    X_input = DataFrame([input_data.dict()])

    # Proces the test data with the process_data function.
    encoder = load_encoder(os.path.join('model', 'encoder_dtc.pkl'))
    lb = load_lb(os.path.join('model', 'lb_dtc.pkl'))

    # Run: process data
    X_infer, _, _, _ = process_data(
        X_input,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        training=False,
    )

    # Run:inference
    pred = inference(model=api_model, X=X_infer)

    # Run: inverse of the binarizer to get: "<=50K" or "">50K"
    return {"Prediction": lb.inverse_transform(pred)[0]}

