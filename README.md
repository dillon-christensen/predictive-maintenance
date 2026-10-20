# Predictive Maintenance Machine Learning Application

This project implements a machine learning application for predicting
potential machine failures using equipment operating measurements. It was
developed as a capstone project using the AI4I 2020 Predictive Maintenance
Dataset.

The application uses a Random Forest classification model and provides a
Streamlit interface for entering machine operating conditions, generating
failure predictions, and viewing dataset and model visualizations.

## Features

The application accepts the following machine characteristics:

- Product type
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

The trained model returns an estimated probability of machine failure. A
40% decision threshold is used to classify an observation as either a
potential machine failure or no predicted machine failure.

The application also includes visualizations of the dataset and model
results and an informational section describing the model and recommended
maintenance practices.

## Machine Learning Model

The final model is a Random Forest classifier trained using the AI4I 2020
Predictive Maintenance Dataset.

The dataset contains 10,000 observations. Machine failures account for only
3.39% of the observations, creating a significant class imbalance.

The final model uses a 40% decision threshold and achieved the following
performance on the held-out test dataset:

- Precision: 61.2%
- Recall: 76.5%
- F1 Score: 68.0%
- ROC-AUC: 0.963

## Project Structure

    capstone/
    ├── data/
    │   └── ai4i2020.csv
    ├── models/
    │   └── random_forest_pipeline.joblib
    ├── notebook/
    │   ├── 01_exploratory_data_analysis.ipynb
    │   ├── 02_model_proof_of_concept.ipynb
    │   ├── 03_finalize_preprocessing.ipynb
    │   ├── 04_post_testing.ipynb
    │   └── 05_final_testing.ipynb
    ├── requirements.txt
    ├── README.md
    └── src/
        └── app.py

## Installation

Python 3 is required to run the project.

It is recommended that the project be run inside a Python virtual
environment.

Create a virtual environment:

    python -m venv .venv

Activate the environment on Linux or macOS:

    source .venv/bin/activate

On Windows:

    .venv\Scripts\activate

Install the required Python packages:

    pip install -r requirements.txt

## Running the Application

From the project root directory, run:

    streamlit run app/app.py

Streamlit will start the application and provide a local URL that can be
opened in a web browser.

## Using the Application

1. Select the product type.
2. Enter the air temperature.
3. Enter the process temperature.
4. Enter the rotational speed.
5. Enter the torque.
6. Enter the tool wear.
7. Select **Predict**.

The application displays the estimated probability of machine failure and
indicates whether the observation exceeds the 40% failure decision
threshold.

The application's visualizations can also be used to examine characteristics
of the dataset and model performance.

## Dataset

This project uses the AI4I 2020 Predictive Maintenance Dataset from the UCI
Machine Learning Repository.

The dataset contains synthetic machine operating data designed to reflect
real-world predictive maintenance scenarios.

The original dataset contains machine identifiers, product information,
operating measurements, a machine failure target, and individual failure
mode indicators.

Only information available as model input is used for prediction. The
individual failure-mode indicators are not used as predictive features.

## Testing

The final model was evaluated against a stratified 20% held-out test set.

Application-level testing was also performed by manually entering known test
observations into the Streamlit interface and comparing the resulting
probabilities and classifications with the results produced by the final
testing notebook.

Testing included:

- A known machine-failure observation
- A known non-failure observation
- A failure observation near the 40% decision threshold

The Streamlit application produced the same probabilities and classifications
as the final model testing notebook for these cases.
