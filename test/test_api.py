"""This module includes API testing functions"""

import requests
from pytest import fixture
from main import app
from fastapi.testclient import TestClient

client = TestClient(app)


@fixture(scope='session')
def metrics():
    """Test the inference path"""
    response = client.post(
        "/model_inference",
        json={
            "model_name": "model_1"})
    assert response.status_code == 200

    return response.json()["precision"], response.json()[
        "recall"], response.json()["fbeta"]


def test_read_root():
    """Test the root path"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Welcome to the API"}


def test_inference_precision(metrics):
    """Test the inference path"""

    assert isinstance(metrics[0], float)


def test_inference_recall(metrics):
    """Test the inference path"""

    assert isinstance(metrics[1], float)


def test_inference_fbeta(metrics):
    """Test the inference path"""

    assert isinstance(metrics[2], float)
