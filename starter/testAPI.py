from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_get_path():
    r = client.get("/")

    assert r.status_code == 200
    assert r.json() == {'Greetings From aakash': 'Welcome To Udacity Project on Model Deployment'}


def test_post_model_inference_zero_class(train_data):
    zero_class = train_data[0]
    r = client.post("/model_inference", json=zero_class.to_dict())

    assert r.status_code == 200
    assert r.json() == {'Prediction': ['<=50K']}


def test_post_model_inference_one_class(train_data):
    one_class = train_data[1]
    r = client.post("/model_inference", json=one_class.to_dict())

    assert r.status_code == 200
    assert r.json() == {'Prediction': ['>50K']}