#!/usr/bin/env -S python3 -i

"""
Script to handle the API code here.
Be able to view the interactive API documentation, powered by Swagger UI,
at http://localhost:8000/docs

Read-in of Person class instance is optional regarding original categorical features.

For production code, set debug on False.
For general FastAPI information, see:
https://fastapi.tiangolo.com/tutorial/
For application setup, see:
https://fastapi.tiangolo.com/advanced/events/
For FastAPI beginner tutorial, start with:
https://fastapi.tiangolo.com/tutorial/first-steps/
For advanced FastAPI example, see:
https://github.com/microsoft/cookiecutter-spacy-fastapi/blob/master/%7B%7Bcookiecutter.project_slug%7D%7D/app/api.py
For testing see:
https://fastapi.tiangolo.com/tutorial/testing/

future toDo:
add a custom exception handler with @app.exception_handler()
see: https://fastapi.tiangolo.com/tutorial/handling-errors/


author: Ilona Brinkmeier
date:   2023-09
"""

###################
# Imports
###################

import logging
import uvicorn
import asyncio
import signal
import os
import sys
import yaml
import numpy as np
import pandas as pd

# needed to run this script alone
MAIN_DIR = os.path.join(os.getcwd(), 'src/')
APP_DIR = os.path.join(MAIN_DIR, 'app/')
sys.path.append(MAIN_DIR)
sys.path.append(os.getcwd())
print(f'sys.path : {sys.path}')

from typing import Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, Body, HTTPException, Response, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas import FeatureLabels, Person
from training.ml.data import clean_data
from training.ml.model import inference
from config import get_config
from slice_performance import load_transformer_artifact, load_final_model_artifact

###################
# Coding
###################

# get logging properties
# info see: https://realpython.com/python-logging-source-code/
logger = logging.getLogger(__name__)

# variable to store artifacts names
ml_components = {}

# read in examples
examples_file = os.path.join(APP_DIR, 'examples_request.yml')
with open(examples_file) as f:
    examples_request = yaml.safe_load(f)


# customised exception
class InferenceNotPossible(HTTPException):
    ''' Raised if inference workflow went wrong '''
    def __init__(self) -> None:
        super().__init__(status_code=404, detail="Client error: Inference not possible")


# Define the signal handler function
def graceful_shutdown(signum, frame) -> None:
    # Perform cleanup tasks here (closing db connections, saving state, ...);
    # e.g. has to be filled, if Person items are stored in a database

    # Set the stop condition when receiving SIGTERM.
    loop = asyncio.get_running_loop()
    stop = loop.create_future()
    loop.add_signal_handler(signal.SIGTERM, stop.set_result, None)

    # Finally, exit the application
    logger.warning("Shutting down the FastAPI US Census app")
    sys.exit(0)


# Register the signal handler for SIGTERM
signal.signal(signal.SIGTERM, graceful_shutdown)


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    ''' Handles transformer and model artifacts for startup and shutdown.

    The coding before the yield will be executed before the application starts taking
    requests, during the startup.
    The coding after the yield will be executed after the application finishes handling requests,
    right before the shutdown.
    '''
    try:
        logging.debug('Read in post-market transformer and model artifacts')
        # load ml components: feature transformer and classifier artifacts
        transformer_artifact = load_transformer_artifact()
        ml_components['transformer_artifact'] = transformer_artifact
        model_artifact = load_final_model_artifact()
        ml_components['model_artifact'] = model_artifact

        yield

        # clean up the ML components and release the resources
        logging.debug('Resource cleaning of transformer and model artifacts')
        ml_components.clear()
    except Exception as e:
        logger.exception("Exit: exception of type %s occurred. Details: %s", type(e).__name__, str(e))
    else:
        txt = 'Handling of transformer and model artifacts was successful during lifespan of FastAPI app.'
        logger.debug(txt)


app = FastAPI(
    title = "Udacity MLOps, Project 3 - Prediction Model for Public US Census Bureau Data",
    description = "Deploying a Binary Classification ML Model on Render with FastAPI; \
                   its inference is about having a salary <=50K or >50K",
    version = "0.1",
    lifespan=lifespan,
    debug = True
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )


@app.get("/")
async def root() -> Response:
    ''' Returns welcome message at root level '''
    response = Response(
        status_code=status.HTTP_200_OK,
        content="Welcome to the Udacity MLOps project 3 and its salary prediction application!"
    )
    return response


@app.get("/feature_labels/{feature_name}")
async def feature_labels(feature_name: FeatureLabels) -> Any:
    ''' Read-in feature values with original label from US census dataset '''
    logging.info("Read-in of feature values from examples_request file started")
    feat_value = examples_request['features_labels'][feature_name]
    return feat_value


@app.post("/predict/")
async def predict(person: Person = Body(..., examples=examples_request['test_examples'])):
    '''
    Returns prediction of test examples about income class, being <=50k or >50k,
    so having a proper response status number 200 in such cases.

    If only a few features are having a wrong value type, the model shall be able to handle
    this properly having an inference result of being an <=50k or >50k item as well.

    If most of the features are missing, a value error shall be thrown with response status number 422.
    '''
    logging.info("Model classification inference started")
    person = person.dict()
    features = np.array(
        [person[f] for f in examples_request['features_labels'].keys()]
    ).reshape(1, -1)

    df = pd.DataFrame(features, columns=examples_request['features_labels'].keys())
    
    df_cleaned = clean_data(df, get_config())
    logger.info('Census cleaned new adult person data with %s features',
                df_cleaned.shape[1])
    logger.info('Its columns are: %s', df_cleaned.columns)

    # cleaning inference case for person dataframe (X = df_cleaned), not training
    X_processed = ml_components['transformer_artifact'].transform(df_cleaned)
    # predict income class
    model = ml_components['model_artifact']
    y_pred = inference(model, X_processed)
    logger.info('Predict post y_pred: %s', y_pred)
    if y_pred not in [0, 1]:
        raise InferenceNotPossible(HTTPException('US census prediction workflow error'))

    pred_class = '>50k' if y_pred == 1 else '<=50k'
    logger.info('income prediction label: %s, salary class: %s', y_pred[0], pred_class)

    content_txt = ''.join(
        ['income prediction label: ', str(y_pred[0]),
         ', ',
         'salary class: ', pred_class]
    )
    response = Response(
        status_code = status.HTTP_200_OK,
        content = content_txt,
    )

    return response


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
