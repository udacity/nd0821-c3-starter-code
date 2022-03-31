import pytest

from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_root_get(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the salary predictor"}


def test_root_wrong_get(client):
    response = client.get("/wrong_url")
    assert response.status_code != 200


def test_predict_salary_less_than_or_equal_to_50_k(client):
    request = {
        "age": 49,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Sales",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital-gain": 6174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "Mexico",
    }
    response = client.post(
        "/predictions",
        json=request,
    )

    assert response.status_code == 200
    assert response.json() == {"predicted salary": "<=50K"}


def test_predict_salary_greater_than_50_k(client):
    request_body = {
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
    response = client.post("/predictions", json=request_body)

    assert response.status_code == 200
    assert response.json() == {"predicted salary": ">50K"}


def test_predict_wrong_post(client):
    request = {
        "age": "49",
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Sales",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
    }
    response = client.post("/predictions", json=request)

    assert response.status_code == 422
