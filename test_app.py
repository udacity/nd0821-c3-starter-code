# import pytest

from fastapi.testclient import TestClient
from http import HTTPStatus
from main import app


client = TestClient(app)


def test_hello1():
    """
    Teste Get hello route
    """
    response = client.get('/')
    # assert response.status_code == HTTPStatus.OK
    assert response.request.method == "GET"


def test_hello2():
    """
    Teste Get hello route
    """
    response = client.get('/')
    # assert response.status_code == HTTPStatus.OK
    assert response.json() == {'response_message': 'Hello', 'status_code': 200}


def test_hello3():
    """
    Teste Get hello route
    """
    response = client.get('/')
    assert response.status_code == HTTPStatus.OK


# @pytest.mark.parametrize('test_input', [
#     ('age', "Age of the person - numerical - int"),
#     ('fnlgt', 'MORE INFO NEEDED - numerical - int'),
#     ('race', 'Race of the person - nominal categorical - str')
# ])
# def test_feature_info_status(test_input: str):
#     response = client.get(f'/feature_info/{test_input}')
#     assert response.status_code == HTTPStatus.OK
#     assert response.request.method == "GET"


# @pytest.mark.parametrize('test_input, expected', [
#     ('age', "Age of the person - numerical - int"),
#     ('fnlgt', 'MORE INFO NEEDED - numerical - int'),
#     ('race', 'Race of the person - nominal categorical - str')
# ])
# def test_feature_info_response(test_input: str, expected: str):
#     """Testing features

#     Args:
#         test_input (np.array): input test
#         expected (np.array): results expected
#     """
#     response = client.get(f'/feature_info/{test_input}')
#     assert response.json() == expected


def test_predict_status1():
    """
    Tests POST route predict.
    """
    data = {"age": 55,
            "workclass": "Private",
            "fnlgt": 77516,
            "education": "Masters",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
            }
    response = client.post("/model", json=data)
    assert response.request.method == "POST"


def test_predict_status2():
    """
    Tests POST route predict.
    """
    data = {"age": 55,
            "workclass": "Private",
            "fnlgt": 77516,
            "education": "Masters",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
            }
    response = client.post("/model", json=data)
    assert response.status_code == HTTPStatus.OK


def test_predict_response():
    data = {
    "age": 55,
    "workclass": "Private",
    "fnlgt": 77516,
    "education": "Masters",
    "education-num": 1,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital-gain": 345234523,
    "capital-loss": 1,
    "hours-per-week": 50,
    "native-country": "United-States"
    }

    response = client.post("/model", json=data)
    # assert response.json()['label'] == 0 or response.json()['label'] == 1
    # assert response.json()['prob'] >= 0 and response.json()['label'] <= 1
    # assert response.json()['salary'] == ' >50k' or response.json()[
        # 'salary'] == ' <=50k'
    assert response.json()["Result"] == 1


# def test_missing_feature_predict():
#     """
#     Testing post route for missing predict.
#     """
#     data = {
#         "age": 0
#     }
#     response = client.post("/model/", json=data)
#     assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
#     assert response.request.method == "POST"
#     assert response.json()["detail"][0]["type"] == "value_error.missing"
