# Check render requests for FastAPI app which is using httpx, based on requests
# see: https://www.python-httpx.org/quickstart/

import httpx


sample = {
    "age": 38,
    "workclass": "Private",
    "fnlgt": 215646,
    "education": "HS-grad",
    "education_num": 9,
    "marital_status": "Divorced",
    "occupation": "Handlers-cleaners",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States",
}

# GET, check first feature attribute
response = httpx.get('https://census-project-xki0.onrender.com/feature_labels/age')
print('--- Render census-project feature GET life service test ---')
print(f'response status code: {response.status_code}')
print(f'response json value: {response.json()}\n')

# POST, check given sample
print('--- Render census-project person data POST life service test ---')
response = httpx.post('https://census-project-xki0.onrender.com/predict/',
                      json=sample)
print(f'response status code: {response.status_code}')
print(f'response content:\n{response.content}')