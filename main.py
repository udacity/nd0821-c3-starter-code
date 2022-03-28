from fastapi import FastAPI

import logging
import os


logging.basicConfig(level=logging.DEBUG, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI(
    title="API for salary predictor",
    description="This API helps to classify",
    version="0.0.1",
)


@app.get("/")
def root():
    return {'message': 'Welcome to the salary predictor'}
