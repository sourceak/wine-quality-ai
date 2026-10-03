import json
from unittest.mock import MagicMock, patch

from src.app import (
    REQUIRED_FEATURES,
    extract_features,
    predict_quality,
    validate_features,
)


COMPLETE_FEATURES = {
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


class FakeModel:
    """Simple fake model used to test the interface without loading ML artifacts."""

    def predict(self, dataframe):
        return [5.25]


def test_extract_features_parses_llm_response(monkeypatch):
    """Natural-language extraction should parse the LLM JSON response."""

    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    fake_response = MagicMock()
    fake_response.output_text = json.dumps(COMPLETE_FEATURES)

    fake_client = MagicMock()
    fake_client.responses.create.return_value = fake_response

    with patch("src.app.OpenAI", return_value=fake_client):
        result = extract_features(
            "I have a red wine with all required measurements."
        )

    assert result == COMPLETE_FEATURES
    fake_client.responses.create.assert_called_once()


def test_complete_parsed_input_can_be_predicted():
    """A complete set of extracted features should reach the model."""

    model = FakeModel()

    result = predict_quality(
        model,
        COMPLETE_FEATURES,
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

    model = MagicMock()

    result = predict_quality(
        model,
        features,
    )

    assert result["success"] is False
    assert result["prediction"] is None
    assert len(result["missing_features"]) > 0
    model.predict.assert_not_called()