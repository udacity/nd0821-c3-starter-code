"""Send one POST request to the live API and print the status code and prediction.

Usage: python live_post.py [API_URL]
The URL can also be set with the API_URL environment variable.
"""
import os
import sys

import requests

API_URL = os.environ.get("API_URL", "https://census-income-api.onrender.com")

person = {
    "age": 46,
    "workclass": "Self-emp-inc",
    "fnlgt": 192779,
    "education": "Prof-school",
    "education-num": 15,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15024,
    "capital-loss": 0,
    "hours-per-week": 60,
    "native-country": "United-States",
}


def main(url: str) -> None:
    # Free Render instances sleep when idle, so the first request can take ~1 minute.
    r = requests.post(f"{url.rstrip('/')}/predict", json=person, timeout=120)
    print(f"Status code: {r.status_code}")
    print(f"Result: {r.json()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else API_URL)
