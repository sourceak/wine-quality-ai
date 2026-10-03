import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

MODEL_PATH = Path("models/best_model.pkl")

REQUIRED_FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "wine_type",
]


# --------------------------------------------------
# Load ML model
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run src/compare_experiments.py first."
        )

    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# Validate extracted features
# --------------------------------------------------

def validate_features(features):
    missing_features = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in features
        or features[feature] is None
    ]

    if missing_features:
        return False, missing_features

    if features["wine_type"] not in ["red", "white"]:
        return False, ["wine_type"]

    return True, []


# --------------------------------------------------
# LLM: natural language -> structured features
# --------------------------------------------------

def extract_features(user_input):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY was not found in the .env file."
        )

    client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)

    prompt = f"""
You extract wine measurements for a machine-learning model.

Extract ONLY information explicitly provided by the user.
Never guess or invent missing values.

Return ONLY valid JSON.

The required fields are:

{json.dumps(REQUIRED_FEATURES, indent=2)}

Rules:
- Numerical fields must be numbers.
- wine_type must be "red" or "white".
- If a value is missing, use null.
- Do not add extra fields.
- Do not estimate missing measurements.
- Do not include markdown.

User message:
{user_input}
"""

    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input=prompt,
    )

    text = response.output_text.strip()

    return json.loads(text)


# --------------------------------------------------
# ML prediction
# --------------------------------------------------

def predict_quality(model, features):
    valid, missing_features = validate_features(features)

    if not valid:
        return {
            "success": False,
            "prediction": None,
            "missing_features": missing_features,
        }

    wine = pd.DataFrame(
        [features],
        columns=REQUIRED_FEATURES,
    )

    prediction = model.predict(wine)[0]

    return {
        "success": True,
        "prediction": float(prediction),
        "missing_features": [],
    }


# --------------------------------------------------
# Streamlit application
# --------------------------------------------------

def main():
    st.set_page_config(
        page_title="Wine Quality AI",
        page_icon="🍷",
    )

    st.title("🍷 Wine Quality AI")

    st.write(
        "Describe a red or white wine using its chemical "
        "measurements. AI will extract the measurements and "
        "use the trained machine-learning model to predict "
        "its quality."
    )

    st.info(
        "The model needs all 11 chemical measurements plus "
        "the wine type (red or white)."
    )

    example = (
        "I have a red wine with fixed acidity 7.4, "
        "volatile acidity 0.70, citric acid 0, "
        "residual sugar 1.9, chlorides 0.076, "
        "free sulfur dioxide 11, total sulfur dioxide 34, "
        "density 0.9978, pH 3.51, sulphates 0.56, "
        "and alcohol 9.4."
    )

    user_input = st.text_area(
        "Describe your wine:",
        placeholder=example,
        height=180,
    )

    if st.button("Predict Quality"):
        if not user_input.strip():
            st.warning("Please describe a wine first.")
            return

        try:
            with st.spinner("Analyzing wine..."):
                features = extract_features(user_input)

            st.subheader("Extracted measurements")
            st.json(features)

            model = load_model()

            result = predict_quality(
                model,
                features,
            )

            if not result["success"]:
                missing = ", ".join(
                    result["missing_features"]
                )

                st.warning(
                    "I don't have enough information to make "
                    "a prediction."
                )

                st.write(
                    f"Missing information: **{missing}**"
                )

                st.write(
                    "Please provide those measurements and try again."
                )

                return

            prediction = result["prediction"]

            st.success(
                f"Predicted wine quality: {prediction:.2f} / 10"
            )

            st.write(
                "This prediction was produced by the trained "
                "Random Forest regression model selected from "
                "the MLflow experiments."
            )

            st.caption(
                "This is a machine-learning estimate based on "
                "physicochemical wine measurements. It should "
                "not be treated as an objective or professional "
                "wine-quality rating."
            )

        except json.JSONDecodeError:
            st.error(
                "The AI could not convert the description into "
                "structured wine measurements. Please try again."
            )

        except Exception as error:
            st.error(f"Application error: {error}")


if __name__ == "__main__":
    main()