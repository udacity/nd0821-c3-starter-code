import requests

# Replace with your actual deployed URL
url = "https://nd0821-c3-starter-code-w71g.onrender.com/inference"

# Example payload (match your Pydantic model structure and alias fields)
payload = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 284582,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Exec-managerial",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

response = requests.post(url, json=payload)

print("Status code:", response.status_code)
print("Response JSON:", response.json())
