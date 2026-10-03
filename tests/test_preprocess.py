import numpy as np

from src.preprocess import (
    build_preprocessor,
    load_data,
    split_features_target,
)


def test_missing_values_are_handled():
    """Preprocessing should impute missing numerical values."""

    wine = load_data()
    X, _ = split_features_target(wine)

    X.loc[0, "fixed acidity"] = np.nan
    X.loc[1, "alcohol"] = np.nan

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    assert not np.isnan(transformed).any()


def test_categorical_feature_is_encoded():
    """wine_type should be converted from text into numerical columns."""

    wine = load_data()
    X, _ = split_features_target(wine)

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    # 11 numerical features + 2 wine-type encoded columns
    assert transformed.shape[1] == 13

    feature_names = preprocessor.get_feature_names_out()

    assert any(
        "wine_type_red" in name
        for name in feature_names
    )

    assert any(
        "wine_type_white" in name
        for name in feature_names
    )


def test_numeric_features_are_scaled():
    """StandardScaler should center numerical features around zero."""

    wine = load_data()
    X, _ = split_features_target(wine)

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    # First 11 output columns are the numerical features.
    numeric_transformed = transformed[:, :11]

    means = numeric_transformed.mean(axis=0)

    assert np.allclose(
        means,
        0,
        atol=1e-7,
    )


def test_original_dataframe_is_not_modified():
    """Preprocessing should not modify the original DataFrame."""

    wine = load_data()
    original = wine.copy(deep=True)

    X, _ = split_features_target(wine)

    preprocessor = build_preprocessor(X)
    preprocessor.fit_transform(X)

    assert wine.equals(original)