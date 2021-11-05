# Put the code for your API here.
from fastapi import FastAPI


app = FastAPI()


@app.get("/ping")
def ping():
    return {"ping": "pong!"}
