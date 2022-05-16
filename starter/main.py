from typing import Optional, Union
import pandas as pd
import os
import yaml

from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml.model import load_model, inference
from starter.ml.data import process_data

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


class CensusItem(BaseModel):
    age: Optional[Union[int, list]] = [39, 52]
    workclass: Optional[Union[str, list]] = ['State-gov', 'Self-emp-inc']
    fnlgt: Optional[Union[int, list]] = [77516, 287927]
    education: Optional[Union[str, list]] = ['Bachelors', 'HS-grad']
    education_num: Optional[Union[int, list]] = Field([13, 9], alias='education-num')
    marital_status: Optional[Union[str, list]] = Field(['Never-married', 'Married-civ-spouse'], alias='marital-status')
    occupation: Optional[Union[str, list]] = ['Adm-clerical', 'Exec-managerial']
    relationship: Optional[Union[str, list]] = ['Not-in-family', 'Wife']
    race: Optional[Union[str, list]] = ['White', 'White']
    sex: Optional[Union[str, list]] = ['Male', 'Female']
    capital_gain: Optional[Union[int, list]] = Field([2174, 15024], alias='capital-gain')
    capital_loss: Optional[Union[int, list]] = Field([0, 0], alias='capital-loss')
    hours_per_week: Optional[Union[int, list]] = Field([40, 40], alias='hours-per-week')
    native_country: Optional[Union[str, list]] = Field(['United-States', 'United-States'], alias='native-country')

    class Config:
        allow_population_by_field_name = True
        

root_path = os.path.dirname(os.path.abspath(__file__))
model = load_model(root_path, 'model.pkl')
preprocessor = load_model(root_path, 'preprocessor.pkl')

with open(os.path.join(root_path, "starter", "constants.yaml"), 'r') as f:
    categorical_features = yaml.safe_load(f)["categorical_features"]

app = FastAPI()

@app.get("/")
async def welcome_message():
    return {'Greetings From Aakash': 'Welcome To Udacity Project on Model Deployment'}


@app.post("/model_inference")
async def model_inference(data: CensusItem):
    data_dic = data.dict(by_alias=True)

    for key, value in data_dic.items():
        if not isinstance(value, list):
            data_dic[key] = [value]

    df = pd.DataFrame(data_dic)

    data_processed, _, _, _ = process_data(
        df, categorical_features=categorical_features, label=None, training=False,
        preprocessor=preprocessor, label_binarizer=None
    )

    pred = list(inference(model, data_processed))

    for idx, val in enumerate(pred):
        if pred[idx] == 0:
            pred[idx] = '<=50K'
        else:
            pred[idx] = '>50K'

    return {'Prediction': pred}




