import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X, categorical_features=None, label=None, training=True, encoder=None, lb=None
):
    """Encode categoricals and labels for training or inference."""
    categorical_features = categorical_features or []
    X = X.copy()
    if label is not None and label in X.columns:
        y = X.pop(label)
    else:
        y = None

    for column in X.select_dtypes(include=["object"]).columns:
        X[column] = X[column].str.strip()
    X_continuous = X.drop(columns=categorical_features).to_numpy(dtype=float)
    X_categorical = X[categorical_features].to_numpy()

    if training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_categorical = encoder.fit_transform(X_categorical)
        lb = LabelBinarizer()
        y = lb.fit_transform(y).ravel() if y is not None else np.array([])
    else:
        if encoder is None:
            raise ValueError("A fitted encoder is required in inference mode.")
        X_categorical = encoder.transform(X_categorical)
        if y is not None:
            if lb is None:
                raise ValueError("A fitted label binarizer is required for labels.")
            y = lb.transform(y).ravel()
        else:
            y = np.array([])

    return np.concatenate([X_continuous, X_categorical], axis=1), y, encoder, lb
