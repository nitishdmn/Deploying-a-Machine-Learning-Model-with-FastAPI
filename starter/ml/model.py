from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


def train_model(X_train, y_train):
    """Fit and return a deterministic random forest classifier."""
    model = RandomForestClassifier(
        n_estimators=150, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """Return binary precision, recall, and F1 metrics."""
    fbeta = fbeta_score(y, preds, beta=1, zero_division=0)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    return precision, recall, fbeta


def inference(model, X):
    """Return predictions for a batch of processed feature rows."""
    return model.predict(X)
