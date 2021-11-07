import logging

from starter.main import CensusRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def test_root(test_app):
    response = test_app.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Hello World!"


def test_ping(test_app):
    response = test_app.get("/ping")
    assert response.status_code == 200
    assert response.json()["ping"] == "pong!"


def test_inference_request_with_prediction_below_50K(test_app):
    body = CensusRequest(
        **{"age": 39, "workclass": "State-gov", "fnlgt": 77516, "education": "Bachelors", "education-num": 13,
           "marital-status": "Never-married", "occupation": "Adm-clerical", "relationship": "Not-in-family",
           "race": "White", "sex": "Male", "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40,
           "native-country": "United-States"})

    response = test_app.post("/predict", data=body.json(by_alias=True))

    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"


def test_inference_request_with_prediction_above_50K(test_app):
    body = CensusRequest(
        **{"age": 31, "workclass": "Private", "fnlgt": 45781, "education": "Masters", "education-num": 14,
           "marital-status": "Never-married", "occupation": "Prof-specialty", "relationship": "Not-in-family",
           "race": "White", "sex": "Female", "capital-gain": 14084, "capital-loss": 0, "hours-per-week": 50,
           "native-country": "United-States"})

    response = test_app.post("/predict", data=body.json(by_alias=True))

    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"
