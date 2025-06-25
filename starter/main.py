from fastapi import FastAPI
from typing import List

import joblib
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field

from starter.starter.ml.data import process_data

app = FastAPI(title="Census Income Prediction API")


@app.get("/")
async def read_root() -> dict:
    """Root endpoint giving a welcome message."""
    return {"message": "Welcome!"}


# ----------------------------------------------------------------------
# Load the trained model once at startup
# ----------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent 
MODEL_DIR = ROOT_DIR / "model"
loaded_model = joblib.load(MODEL_DIR / "model.pkl")
loaded_encoder = joblib.load(MODEL_DIR / "encoder.pkl")
loaded_lb = joblib.load(MODEL_DIR / "label_binarizer.pkl")


# ----------------------------------------------------------------------
# Request body schema using Pydantic
# ----------------------------------------------------------------------
class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    model_config = {
         # Lets clients send either naming style (education_num or education-num).
        "populate_by_name": True, 
        "json_schema_extra": {
            "examples": [
                {
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
                    "native-country": "United-States",
                }
            ]
        },
    }


@app.post("/predict")
async def predict(data: CensusData) -> dict:
    """Return model predictions for a single Census record."""
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

    # Convert incoming data to DataFrame with the column names expected by the model
    input_df = pd.DataFrame([data.model_dump(by_alias=True)])

    # Perform inference. The loaded model is assumed to include any preprocessing
    X_inf, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        label="salary",
        training=False,         
        encoder=loaded_encoder,
        lb=loaded_lb,
    )
    preds = loaded_model.predict(X_inf)
    print(f"Predictions: {preds}")

    # For binary classification with LabelBinarizer we map 0/1 back to string labels.
    pred_label: List[str]
    try:
        # If model uses label binarizer like in training pipeline we can access classes_
        classes = getattr(loaded_model, "classes_", None)
        if classes is not None:
            pred_label = [classes[preds[0]]]
        else:
            pred_label = [str(preds[0])]
    except Exception:
        pred_label = [str(preds[0])]

    return {"prediction": pred_label[0]}


# --- Entry point --------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
