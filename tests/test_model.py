from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.preprocess import load_data, split_features_target


MODEL_PATH = Path("models/best_model.pkl")


def test_model_prediction_shape_and_type():
    """The saved model should return one numeric prediction per sample."""

    model = joblib.load(MODEL_PATH)

    wine = load_data()
    X, _ = split_features_target(wine)

    sample = X.iloc[:5]

    predictions = model.predict(sample)

    assert predictions.shape == (5,)
    assert np.issubdtype(predictions.dtype, np.number)
    assert np.isfinite(predictions).all()


def test_model_meets_performance_threshold():
    """The best model should meet minimum performance requirements."""

    model = joblib.load(MODEL_PATH)

    wine = load_data()
    X, y = split_features_target(wine)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    predictions = model.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    # Conservative thresholds below the model's observed
    # RMSE ~0.608 and R² ~0.500.
    assert rmse < 0.80
    assert r2 > 0.30