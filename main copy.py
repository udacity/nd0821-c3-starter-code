# Put the code for your API here.
import os
import pickle as pkl
import pandas as pd
import uvicorn
from fastapi.responses import JSONResponse
from fastapi import FastAPI

from data_model import BasicInputData
import starter.config as config
from starter.ml.data import process_data
from starter.ml.model import inference


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


app = FastAPI(
    title="API for Salary Model",
    description="Return prediction for salary",
    version="0.0.1",
)
with open(config.MODEL_PATH, 'rb') as f:
    encoder, lb, model = pkl.load(f)


@app.get("/")
async def welcome():
    """
    Example function for returning home directory.
    Args:
    Returns:
        example_message (Dict) : Example message response for home directory
        GET request.
    """
    return {'message': 'Hello'}

@app.post("/model")
async def prediction(input_data: BasicInputData):
    """
    Example function for returning model output from POST request.
    The function take in a single web form entry and converts it to a single
    row of input data conforming to the constraints of the features used in the model.
    Args:
        input_data (BasicInputData) : Instance of a BasicInputData object. Collected data from
        web form submission.
    Returns:
        json_res (JSONResponse) : A JSON serialized response dictionary containing
        model classification of input data.
    """
    # Formatting input_data
    input_df = pd.DataFrame(
        {k: v for k, v in input_data.dict().items()}, index=[0]
    )

    input_df.columns = [_.replace('_', '-') for _ in input_df.columns]


    x_data, _, _, _ = process_data(
        X=input_df,
        categorical_features=config.cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # get predictions and return
    pred = inference(model, x_data)
    return JSONResponse({"Result": int(pred[0])})