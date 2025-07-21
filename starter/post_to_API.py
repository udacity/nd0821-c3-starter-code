import requests
import json

#url = "enter heroku web app url here"
url = "https://udacity-fastapi-app.herokuapp.com/inference"


# explicit the sample to perform inference on
sample =  { 'age':35,
            'workclass':"Self-emp-not-inc", 
            'fnlgt':120000,
            'education':"Bachelors",
            'education_num':13,
            'marital_status':"Married-civ-spouse",
            'occupation':"Tech-support",
            'relationship':"Husband",
            'race':"White",
            'sex':"Male",
            'capital_gain':5000,
            'capital_loss':0,
            'hours_per_week':40,
            'native_country':"Canada"
            }

data = json.dumps(sample)

# post to API and collect response
response = requests.post(url, data=data )

# display output - response will show sample details + model prediction added
print("response status code", response.status_code)
print("response content:")
print(response.json())