import pandas as pd
from fastapi.testclient import TestClient

from main import app
from starter.train_model import CAT_FEATURES, DATA_PATH
from main import load_artifacts
from starter.ml.data import process_data
from starter.ml.model import inference

client = TestClient(app)


def test_get_root_status_and_body():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Census Income Classifier API."
    }


def _example_for(prediction):
    model, encoder, lb = load_artifacts()
    data = pd.read_csv(DATA_PATH, skipinitialspace=True).head(3000)
    for column in data.select_dtypes(include=["object"]).columns:
        data[column] = data[column].str.strip()
    features, _, _, _ = process_data(
        data.drop(columns=["salary"]), CAT_FEATURES, training=False,
        encoder=encoder, lb=lb,
    )
    indices = (inference(model, features) == prediction).nonzero()[0]
    assert len(indices), f"No example found for prediction {prediction}"
    return data.iloc[int(indices[0])].drop(labels=["salary"]).to_dict()


def test_post_returns_at_most_50k_prediction():
    response = client.post("/predict", json=_example_for(0))
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}


def test_post_returns_over_50k_prediction():
    response = client.post("/predict", json=_example_for(1))
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}


def test_favicon_request_is_handled_without_not_found():
    response = client.get("/favicon.ico")
    assert response.status_code == 204
    assert response.content == b""
