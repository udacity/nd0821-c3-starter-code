import requests
import json
import os

from starter.ml.data import load_data


root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
test_data = load_data(root_path, 'test_census.csv')

test_data = test_data.iloc[0]
label = test_data['salary']

test_data.drop('salary')

response = requests.post('https://udacity-census-inference-app.herokuapp.com/model_inference',
                         data=json.dumps(test_data.to_dict()))

print(f"Status Code: {response.status_code}")
print(f"Ground Truth: {label}, Response: {response.json()}")
