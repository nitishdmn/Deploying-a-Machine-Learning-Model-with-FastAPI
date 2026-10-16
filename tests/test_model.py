import numpy as np
import pandas as pd

from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference, train_model


CATEGORICAL = ["kind"]


def test_process_data_returns_encoded_features_and_fitted_transformers():
    frame = pd.DataFrame({"value": [1, 2, 3, 4], "kind": ["a", "b", "a", "b"], "label": ["n", "y", "n", "y"]})
    features, labels, encoder, lb = process_data(frame, CATEGORICAL, "label")
    assert features.shape == (4, 3)
    assert labels.tolist() == [0, 1, 0, 1]
    assert encoder is not None and lb is not None


def test_train_model_and_inference_return_predictions():
    model = train_model(np.array([[0], [1], [2], [3]]), np.array([0, 0, 1, 1]))
    result = inference(model, np.array([[0], [3]]))
    assert result.shape == (2,)
    assert set(result).issubset({0, 1})


def test_compute_model_metrics_returns_precision_recall_f1():
    assert compute_model_metrics(np.array([0, 1, 1]), np.array([0, 1, 0])) == (1.0, 0.5, 2 / 3)
