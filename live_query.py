import requests


def get_model_inference():
    url = "https://ml-ci-cd-project.herokuapp.com/model_inference"
    response = requests.post(url, json={"model_name": "model_test"})
    if response.status_code == 200:
        return {
            "Status Code": response.status_code,
            "Precision": response.json()["precision"],
            "Recall": response.json()["recall"],
            "Fbeta": response.json()["fbeta"]}
    else:
        return {"Status Code": response.status_code}


if __name__ == "__main__":
    print(get_model_inference())
