from fastapi.testclient import TestClient
from main import app


client = TestClient(app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() ==  {"message": "Welcome to Your Sentiment Classification FastAPI"}
    
def test_predict():
    r = client.post("/slice_test", json = {"model_path":"model/Logic_reg.pkl","data_path":"data/census.csv","feature":"education"})
    print(r.status_code)
    assert r.status_code == 200
    assert r.json() == {"feature slice tested": "education"}
def test_predict_bad_feature():
    try:
        r = client.post("/slice_test",json={"model_path":"model/Logic_reg.pkl","data_path":"data/census.csv","feature":"udacity"})
    except KeyError as err:
        return None 