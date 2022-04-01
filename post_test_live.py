import requests
data = {
        "age": 49,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Sales",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 7174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "Mexico",
    }
request2 = requests.post(f"https://census-app-project-3.herokuapp.com/predictions", json=data)

assert request2.status_code == 200
print(request2.json())