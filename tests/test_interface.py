from src.app import REQUIRED_FEATURES, predict_quality, validate_features


class FakeModel:
    """Simple fake model used to test the interface without loading ML artifacts."""

    def predict(self, dataframe):
        return [5.25]


def test_complete_parsed_input_can_be_predicted():
    """A complete set of extracted features should reach the model."""

    features = {
        "fixed acidity": 7.4,
        "volatile acidity": 0.70,
        "citric acid": 0.00,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
        "wine_type": "red",
    }

    model = FakeModel()

    result = predict_quality(
        model,
        features,
    )

    assert result["success"] is True
    assert result["prediction"] == 5.25
    assert result["missing_features"] == []


def test_incomplete_input_is_handled_gracefully():
    """Incomplete extracted input should not be sent to the model."""

    features = {
        "alcohol": 9.4,
        "wine_type": "red",
    }

    valid, missing_features = validate_features(features)

    assert valid is False
    assert len(missing_features) > 0

    for feature in missing_features:
        assert feature in REQUIRED_FEATURES

    model = FakeModel()

    result = predict_quality(
        model,
        features,
    )

    assert result["success"] is False
    assert result["prediction"] is None
    assert len(result["missing_features"]) > 0