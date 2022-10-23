"""This module includes API testing functions"""

from main import app
from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def client():
    """
    Get dataset
    """
    api_client = TestClient(app)
    return api_client


def test_get(client):
    """Test that the API is up and running (root)"""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Greetings to our API!"}


def test_get_malformed(client):
    """Test that the API returns not successful (root)"""
    r = client.get("/wrong_url")
    assert r.status_code != 200


def test_post_above(client):
    """Test post request with data resulting in prediction above 50K"""
    r = client.post("/", json={
        "age": 60,
        "workclass": "Private",
        "education": "Bachelors",
        "maritalStatus": "Separated",
        "occupation": "Sales",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Female",
        "hoursPerWeek": 40,
        "nativeCountry": "United-States"
    })
    assert r.status_code == 200
    assert r.json() == {"prediction": " >50K"}


def test_post_below(client):
    """Test post request with data resulting in prediction below 50K"""
    r = client.post("/", json={
        "age": 19,
        "workclass": "Private",
        "education": "Doctorate",
        "maritalStatus": "Never-married",
        "occupation": "Armed-Forces",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Male",
        "hoursPerWeek": 20,
        "nativeCountry": "United-States"
    })
    assert r.status_code == 200
    assert r.json() == {"prediction": " <=50K"}


def test_post_malformed(client):
    """Test post request with malformed data"""
    r = client.post("/", json={
        "age": 32,
        "workclass": "Private",
        "education": "Some-college",
        "maritalStatus": "ERROR",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male",
        "hoursPerWeek": 60,
        "nativeCountry": "United-States"
    })
    assert r.status_code == 422
