"""This module includes API testing functions"""

import requests


def test_read_root():
    """Test the root path"""
    response = requests.get("http://localhost:8000/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Welcome to the API"}


def test_inference_precision():
    """Test the inference path"""
    response = requests.post(
        "http://localhost:8000/model_inference",
        json={
            "model_name": "model_1"})
    assert response.status_code == 200
    assert isinstance(response.json()["precision"], float)


def test_inference_recall():
    """Test the inference path"""
    response = requests.post(
        "http://localhost:8000/model_inference",
        json={
            "model_name": "model_1"})
    assert response.status_code == 200
    assert isinstance(response.json()["recall"], float)


def test_inference_fbeta():
    """Test the inference path"""
    response = requests.post(
        "http://localhost:8000/model_inference",
        json={
            "model_name": "model_1"})
    assert response.status_code == 200
    assert isinstance(response.json()["fbeta"], float)
