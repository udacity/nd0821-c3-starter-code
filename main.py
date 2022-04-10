# Put the code for your API here.
import pandas as pd
import pickle as pkl
import os
import starter.config as config
from starlette.status import HTTP_200_OK

from data_model import BasicInputData, BasicInputDataPost
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from starter.ml.data import process_data
from starter.ml.model import inference


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


app = FastAPI(
    title="API for Salary Model",
    description="Return prediction for salary.",
    version="1.0.0",
)
with open(config.MODEL_PATH, 'rb') as f:
    encoder, lb, model = pkl.load(f)


@app.get("/", status_code=HTTP_200_OK,
         response_model=BasicInputDataPost, summary="Teste Get.",
         description='Testa de a api está em pé via GET.')
async def hello() -> dict:
    """
        Rota para verificação, apenas um get.
    """

    return {'response_message': 'Hello', 'status_code': 200}


@app.post("/model")
async def prediction(input_data: BasicInputData):
    """
    Example function for returning model output from
    POST request.
    The function take in a single web form entry and
    converts it to a single
    row of input data conforming to the constraints of
    the features used in the model.
    Args:
        input_data (BasicInputData) : Instance of a
        BasicInputData object. Collected data from
        web form submission.
    Returns:
        json_res (JSONResponse) : A JSON serialized
        response dictionary containing
        model classification of input data.
    """
    # Formatting input_data
    input_df = pd.DataFrame(
        {k: v for k, v in input_data.dict().items()}, index=[0]
    )

    input_df.columns = [_.replace('_', '-') for _ in input_df.columns]

    x_data, _, _, _ = process_data(
        X=input_df,
        categorical_features=config.cat_feat,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # get predictions and return
    pred = inference(model, x_data)
    return JSONResponse({"Result": int(pred[0])})
