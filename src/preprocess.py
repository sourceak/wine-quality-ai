from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RED_WINE_PATH = Path("data/winequality-red.csv")
WHITE_WINE_PATH = Path("data/winequality-white.csv")

TARGET_COLUMN = "quality"
CATEGORICAL_FEATURES = ["wine_type"]


def load_data(
    red_path=RED_WINE_PATH,
    white_path=WHITE_WINE_PATH,
):
    """Load and combine the red and white wine datasets."""

    red_wine = pd.read_csv(red_path, sep=";")
    white_wine = pd.read_csv(white_path, sep=";")

    red_wine["wine_type"] = "red"
    white_wine["wine_type"] = "white"

    wine = pd.concat(
        [red_wine, white_wine],
        ignore_index=True,
    )

    return wine


def split_features_target(df):
    """Separate model features from the target."""

    X = df.drop(columns=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def build_preprocessor(X):
    """Create the preprocessing pipeline."""

    numeric_features = [
        column
        for column in X.columns
        if column not in CATEGORICAL_FEATURES
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


if __name__ == "__main__":
    wine = load_data()
    X, y = split_features_target(wine)
    preprocessor = build_preprocessor(X)

    X_transformed = preprocessor.fit_transform(X)

    print("Original dataset:", wine.shape)
    print("Features:", X.shape)
    print("Target:", y.shape)
    print("Transformed features:", X_transformed.shape)