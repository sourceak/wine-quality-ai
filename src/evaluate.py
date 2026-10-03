from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/best_model.pkl")


def main():
    # Load the saved model pipeline
    model = joblib.load(MODEL_PATH)

    # Example wine
    wine = pd.DataFrame(
        [
            {
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
        ]
    )

    prediction = model.predict(wine)

    print("Wine:")
    print(wine.to_string(index=False))

    print("\nPredicted quality:")
    print(f"{prediction[0]:.2f}")


if __name__ == "__main__":
    main()