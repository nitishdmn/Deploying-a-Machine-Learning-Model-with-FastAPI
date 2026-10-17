"""Train the census classifier, save artifacts, and report slice metrics."""
from pathlib import Path
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference, train_model

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "census.csv"
MODEL_DIR = ROOT / "model"
CAT_FEATURES = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
LABEL = "salary"


def compute_slice_metrics(
    data, model, encoder, lb, categorical_features, feature, label=LABEL
):
    """Compute precision, recall, and F1 for every value of a feature.

    The data must contain the ground-truth label. Preprocessing uses the
    already-fitted encoder and label binarizer from the training split.
    """
    if feature not in data.columns:
        raise ValueError(f"Slice feature {feature!r} is not in the data")
    if label not in data.columns:
        raise ValueError(f"Target column {label!r} is not in the data")

    rows = []
    for value in sorted(data[feature].dropna().unique()):
        subset = data[data[feature] == value]
        X_slice, y_slice, _, _ = process_data(
            subset, categorical_features=categorical_features, label=label,
            training=False, encoder=encoder, lb=lb,
        )
        precision, recall, f1 = compute_model_metrics(
            y_slice, inference(model, X_slice)
        )
        rows.append((value, precision, recall, f1))
    return rows


def main():
    data = pd.read_csv(DATA_PATH, skipinitialspace=True)
    for column in data.select_dtypes(include=["object"]).columns:
        data[column] = data[column].str.strip()
    train, test = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data[LABEL]
    )
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label=LABEL, training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label=LABEL, training=False,
        encoder=encoder, lb=lb,
    )
    model = train_model(X_train, y_train)
    MODEL_DIR.mkdir(exist_ok=True)
    artifacts = (
        ("model.pkl", model),
        ("encoder.pkl", encoder),
        ("label_binarizer.pkl", lb),
    )
    for name, artifact in artifacts:
        with (MODEL_DIR / name).open("wb") as file:
            pickle.dump(artifact, file)

    precision, recall, f1 = compute_model_metrics(
        y_test, inference(model, X_test)
    )
    print(
        f"Test metrics: precision={precision:.4f}, "
        f"recall={recall:.4f}, f1={f1:.4f}"
    )
    lines = ["Slice metrics by education", "education,precision,recall,f1"]
    for value, p, r, f in compute_slice_metrics(
        test, model, encoder, lb, CAT_FEATURES, feature="education"
    ):
        lines.append(f"{value},{p:.4f},{r:.4f},{f:.4f}")
    (ROOT / "slice_output.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
