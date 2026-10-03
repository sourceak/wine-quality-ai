# Wine Quality AI

Wine Quality AI is an end-to-end machine learning application that predicts wine quality from physicochemical measurements.

The project combines a trained regression model, MLflow experiment tracking, an LLM-powered natural-language interface, automated testing, and a Streamlit web application.

Instead of requiring users to manually construct model inputs, the application allows them to describe a wine in natural language. An LLM extracts the required measurements, validates them, and passes them to the trained machine learning pipeline.

## Problem

Wine quality prediction normally requires structured numerical data. This project provides a more accessible interface where a user can describe a wine using natural language.

For example:

> I have a red wine with fixed acidity 7.4, volatile acidity 0.70, citric acid 0, residual sugar 1.9, chlorides 0.076, free sulfur dioxide 11, total sulfur dioxide 34, density 0.9978, pH 3.51, sulphates 0.56, and alcohol 9.4.

The application extracts the measurements and predicts the wine's quality.

## Dataset

The project uses the UCI Wine Quality dataset containing red and white Portuguese vinho verde wines.

The combined dataset contains:

- 6,497 samples
- 11 physicochemical measurements
- Wine type (`red` or `white`)
- Wine quality score as the regression target

The red dataset contains 1,599 samples and the white dataset contains 4,898 samples.

## Features

The model uses:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Wine type

The target variable is `quality`.

## Architecture

```text
Natural-language description
            |
            v
        LLM parser
            |
            v
Structured wine measurements
            |
            v
     Input validation
            |
            v
 Preprocessing pipeline
            |
            v
 Random Forest Regressor
            |
            v
   Quality prediction
            |
            v
     Streamlit UI
```

The preprocessing pipeline performs:

- Median imputation for missing numerical values
- Most-frequent imputation for categorical values
- Standard scaling of numerical features
- One-hot encoding of wine type

The preprocessing pipeline and regression model are saved together so inference uses exactly the same transformations used during training.

## Model Experiments

Five model configurations were trained and tracked with MLflow.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.5644 | 0.7357 | 0.2672 |
| Random Forest 1 | 0.4736 | 0.6298 | 0.4630 |
| Random Forest 2 | **0.4385** | **0.6079** | **0.4996** |
| Gradient Boosting 1 | 0.5365 | 0.6845 | 0.3655 |
| Gradient Boosting 2 | 0.5220 | 0.6683 | 0.3952 |

The best model is selected programmatically from successful MLflow runs using RMSE.

The selected model is **Random Forest 2**.

### Best Model Performance

- MAE: 0.4385
- RMSE: 0.6079
- R²: 0.4996

## MLflow

MLflow is used to track:

- Model type
- Hyperparameters
- Dataset information
- Training/test sample counts
- MAE
- RMSE
- R²
- Trained model artifacts

The project uses `mlflow.search_runs()` to compare completed experiments and automatically identify the run with the lowest RMSE.

To open the MLflow interface:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

## LLM Interface

The application uses an LLM through Groq's OpenAI-compatible API.

The LLM does not predict wine quality itself.

Its job is to convert a natural-language description into the structured features required by the machine learning model.

The application explicitly instructs the LLM not to invent missing measurements.

If required information is missing, the application identifies the missing features instead of making a prediction.

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd wine-quality-ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the LLM API key

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Never commit the `.env` file.

## Running the Application

Start the Streamlit application:

```bash
streamlit run src/app.py
```

Then open the local URL displayed by Streamlit.

Enter a natural-language wine description and select **Predict Quality**.

## Training

Train all configured models with:

```bash
python src/train.py
```

Training parameters are stored in:

```text
configs/config.yaml
```

The training script logs each experiment to MLflow.

## Selecting the Best Model

Run:

```bash
python src/compare_experiments.py
```

The script:

1. Searches successful MLflow runs.
2. Compares MAE, RMSE, and R².
3. Selects the model with the lowest RMSE.
4. Loads the winning MLflow model.
5. Saves it as `models/best_model.pkl`.

## Evaluation

A saved model can be tested directly with:

```bash
python src/evaluate.py
```

For the example wine included in the evaluation script, the selected model predicts a quality score of approximately:

```text
5.02
```

## Testing

The project contains automated tests for preprocessing, model behavior, performance, and interface validation.

Run all tests with:

```bash
pytest tests/ -v
```

Current test suite:

```text
8 passed
```

The tests cover:

- Missing-value handling
- Categorical encoding
- Numerical scaling
- DataFrame immutability
- Prediction shape and type
- Minimum model performance
- Complete interface input
- Graceful handling of incomplete input

## Project Structure

```text
wine-quality-ai/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── winequality-red.csv
│   └── winequality-white.csv
│
├── models/
│   └── best_model.pkl
│
├── src/
│   ├── app.py
│   ├── compare_experiments.py
│   ├── evaluate.py
│   ├── preprocess.py
│   └── train.py
│
├── tests/
│   ├── test_interface.py
│   ├── test_model.py
│   └── test_preprocess.py
│
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

Raw datasets, trained model artifacts, environment variables, and local MLflow artifacts are excluded from Git where appropriate.

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- MLflow
- Streamlit
- Groq
- OpenAI-compatible Python SDK
- pytest
- YAML
- joblib

## Reflection

This project demonstrates how a traditional machine learning model can be integrated with a modern LLM interface without using the LLM as a replacement for the predictive model.

The machine learning pipeline remains responsible for the wine-quality prediction, while the LLM provides a natural-language interface for translating user input into structured features.

Separating preprocessing, training, experiment tracking, inference, LLM parsing, and testing makes the application easier to evaluate, maintain, and extend.

One limitation is that the model requires all physicochemical measurements before making a prediction. A future version could support partial-input models, uncertainty estimates, additional model tuning, and deployment to a hosted environment.