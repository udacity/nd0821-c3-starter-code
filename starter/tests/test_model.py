import logging
import starter.starter.ml.model as model
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def test_can_train_model(get_process_data):
    X, y, encoder, lb = get_process_data
    ml_model = model.train_model(X_train=X, y_train=y)
    assert ml_model is not None
    assert isinstance(ml_model, LogisticRegression)


def test_can_compute_model_metrics(get_process_data):
    X, y, encoder, lb = get_process_data
    ml_model = model.train_model(X_train=X, y_train=y)
    preds = ml_model.predict(X)
    precision, recall, fbeta = model.compute_model_metrics(y=y, preds=preds)
    assert isinstance(precision, float),  'precision'
    assert isinstance(recall, float), 'recall'
    assert isinstance(fbeta, float), 'f1'
