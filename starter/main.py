# Put the code for your API here.
import fastapi
from pydantic import BaseModel
from typing import Optional
from starter.train_model import train_eval_model

app = fastapi.FastAPI()


class Item(BaseModel):
    model_name: str  # This is the current model tag.
    description: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "model_name": "model_1",
                "description": "First model"
            }
        }


@app.get("/")
def read_root():
    return {"msg": "Welcome to the API"}


@app.post("/model_inference")
def inference(item: Item):
    # train_model and evaluate the model
    print(" Training model...")
    precision, recall, fbeta = train_eval_model(item.model_name)
    return {
        "model_name": item.model_name,
        "precision": precision,
        "recall": recall,
        "fbeta": fbeta}
