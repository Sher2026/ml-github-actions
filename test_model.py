import pytest
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from train import train_model


@pytest.fixture(scope="module")
def trained_artifacts():
    """Train once and share results across all tests in this file."""
    # Unpack all 5 return values
    artifact, X_test, y_test, accuracy, report = train_model()
    return {
        "model": artifact["model"],
        "X_test": X_test,
        "y_test": y_test,
        "accuracy": accuracy,
    }


def test_model_trains_successfully(trained_artifacts):
    assert trained_artifacts["model"] is not None
    assert trained_artifacts["accuracy"] >= 0.80


def test_predictions_have_correct_length(trained_artifacts):
    model = trained_artifacts["model"]
    X_test = trained_artifacts["X_test"]
    y_test = trained_artifacts["y_test"]

    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)


def test_predictions_are_valid_classes(trained_artifacts):
    model = trained_artifacts["model"]
    X_test = trained_artifacts["X_test"]

    predictions = model.predict(X_test)
    valid_classes = set(load_iris().target)
    assert set(predictions).issubset(valid_classes)


def test_accuracy_is_valid(trained_artifacts):
    model = trained_artifacts["model"]
    X_test = trained_artifacts["X_test"]
    y_test = trained_artifacts["y_test"]

    predictions = model.predict(X_test)
    actual_accuracy = accuracy_score(y_test, predictions)

    assert 0.0 <= actual_accuracy <= 1.0
    assert actual_accuracy >= 0.80