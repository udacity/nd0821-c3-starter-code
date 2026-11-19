import io

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    # "with" triggers the lifespan handler, so the model gets loaded
    with TestClient(app) as c:
        yield c


def make_csv(row: dict) -> str:
    header = ",".join(row.keys())
    values = ",".join(str(v) for v in row.values())
    return f"{header}\n{values}\n"


def post_row(client, row: dict):
    csv_text = make_csv(row)
    return client.post(
        "/predict",
        files={
            "file": ("data.csv", io.BytesIO(csv_text.encode()), "text/csv")
        },
    )


# Rows chosen so the model gives a clear-cut answer.
# Extend both with every column your Record/model needs.
BASE_ROWS = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
    "salary": "<=50K",
}
LOW_INCOME_ROW = BASE_ROWS | {
    "age": 22,
    "workclass": "Private",
    "education-num": 9,
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 20,
}

HIGH_INCOME_ROW = BASE_ROWS | {
    "age": 45,
    "workclass": "Self-emp-inc",
    "education-num": 15,
    "capital-gain": 99999,  # very strong ">50K" signal in the census data
    "capital-loss": 0,
    "hours-per-week": 60,
}


def test_get_root(client):
    response = client.get("/")
    assert response.status_code == 200
    # Replace with whatever your root endpoint actually returns
    assert response.json() == {
        "message": "Hello and welcome to simple inference API"
    }


def test_predict_low_income(client):
    response = post_row(client, LOW_INCOME_ROW)
    assert response.status_code == 200

    body = response.json()
    assert len(body["predictions"]) == 1
    assert body["predictions"][0] == "<=50K"


def test_predict_high_income(client):
    response = post_row(client, HIGH_INCOME_ROW)
    assert response.status_code == 200

    body = response.json()
    assert len(body["predictions"]) == 1
    assert body["predictions"][0] == ">50K"


def test_predict_missing_column_returns_422(client):
    row = {k: v for k, v in LOW_INCOME_ROW.items() if k != "age"}
    response = post_row(client, row)
    assert response.status_code == 422


def test_predict_invalid_value_returns_422(client):
    row = {**LOW_INCOME_ROW, "age": "not-a-number"}
    response = post_row(client, row)
    assert response.status_code == 422

