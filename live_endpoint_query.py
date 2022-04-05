"""
Example script for querying the live API hosted on
Heroku.

"""
import os
import requests

URL = "https://award40-udacity.herokuapp.com"

response = requests.post(os.path.join(URL, "/model"), json={
                    "age": 55,
                    "workclass": "Private",
                    "fnlgt": 77516,
                    "education": "Masters",
                    "education-num": 13,
                    "marital-status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Female",
                    "capital-gain": 2174,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States",
            })

print(response.status_code)
print(response.json())