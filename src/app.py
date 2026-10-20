import csv
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FAILURE_THRESHOLD = 0.40

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_pipeline.joblib"
DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"
LOG_FILE = BASE_DIR / "logs" / "predictions.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log_prediction(probability, classification):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    new_file = not LOG_FILE.exists()

    with LOG_FILE.open("a", newline="") as file:
        writer = csv.writer(file)

        if new_file:
            writer.writerow([
                "timestamp",
                "failure_probability",
                "classification"
            ])

        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            round(probability, 4),
            classification
        ])



# ---------------------------------------------------------------------------
# Data and model loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load the trained preprocessing and Random Forest pipeline."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    """Load the AI4I predictive maintenance dataset."""
    return pd.read_csv(DATA_PATH)


model = load_model()
data = load_data()


# ---------------------------------------------------------------------------
# Prediction visualization
# ---------------------------------------------------------------------------

def display_failure_probability(failure_probability):
    """Display the predicted failure probability and decision threshold."""

    st.subheader("Failure Probability")

    probability_data = pd.DataFrame({
        "Measure": [
            "Estimated Failure Probability",
            "Decision Threshold"
        ],
        "Probability": [
            failure_probability,
            FAILURE_THRESHOLD
        ]
    })

    probability_chart = (
        alt.Chart(probability_data)
        .mark_bar()
        .encode(
            x=alt.X(
                "Probability:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%"),
                title="Probability"
            ),
            y=alt.Y(
                "Measure:N",
                title=None,
                sort=[
                    "Estimated Failure Probability",
                    "Decision Threshold"
                ],
                axis=alt.Axis(labelLimit=250)
            ),
            tooltip=[
                alt.Tooltip("Measure:N", title="Measure"),
                alt.Tooltip(
                    "Probability:Q",
                    title="Probability",
                    format=".1%"
                )
            ]
        )
        .properties(height=120)
    )

    st.altair_chart(
        probability_chart,
        width="stretch"
    )

    st.caption(
        f"The model classifies an observation as a potential machine "
        f"failure when its estimated failure probability reaches the "
        f"{FAILURE_THRESHOLD:.0%} decision threshold."
    )


# ---------------------------------------------------------------------------
# Prediction interface
# ---------------------------------------------------------------------------

def display_prediction_form():
    """Display equipment inputs and generate a machine failure prediction."""

    st.header("Equipment Measurements")

    st.write(
        "Enter the current equipment measurements to estimate the "
        "potential for machine failure."
    )

    st.caption(
        "Input values are limited to the ranges represented in the "
        "model's training dataset."
    )

    product_type = st.selectbox(
        "Product Type",
        ["L", "M", "H"]
    )

    air_temperature = st.number_input(
        "Air Temperature [K]",
        min_value=295.3,
        max_value=304.5,
        value=300.0,
        step=0.1
    )

    process_temperature = st.number_input(
        "Process Temperature [K]",
        min_value=305.7,
        max_value=313.8,
        value=310.0,
        step=0.1
    )

    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=1168,
        max_value=2886,
        value=1500,
        step=1
    )

    torque = st.number_input(
        "Torque [Nm]",
        min_value=3.8,
        max_value=76.6,
        value=40.0,
        step=0.1
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=253,
        value=100,
        step=1
    )

    if st.button("Predict", type="primary"):

        # Equiment measurement validation
        if process_temperature < air_temperature:
            st.error(
                "Process temperature cannot be lower than air temperature."
            )
            st.stop()

        if rotational_speed <= 0:
            st.error("Rotational speed must be greater than zero.")
            st.stop()

        if torque < 0 or tool_wear < 0:
            st.error("Torque and tool wear cannot be negative.")
            st.stop()

        input_data = pd.DataFrame({
            "Type": [product_type],
            "Air temperature [K]": [air_temperature],
            "Process temperature [K]": [process_temperature],
            "Rotational speed [rpm]": [rotational_speed],
            "Torque [Nm]": [torque],
            "Tool wear [min]": [tool_wear]
        })

        failure_probability = model.predict_proba(input_data)[0, 1]

        if failure_probability >= FAILURE_THRESHOLD:
            st.error("Potential Machine Failure")
        else:
            st.success("No Machine Failure Predicted")

        st.write(
            f"Estimated failure probability: "
            f"**{failure_probability:.1%}**"
        )

        display_failure_probability(failure_probability)

        # Log Prediction
        log_prediction(
            failure_probability,
            "Potential Machine Failure"
            if failure_probability >= FAILURE_THRESHOLD
            else "Normal Operation"
        )


# ---------------------------------------------------------------------------
# Dataset visualizations
# ---------------------------------------------------------------------------

def display_failure_distribution():
    """Display the distribution of failure and non-failure observations."""

    st.subheader("Machine Failure Distribution")

    failure_counts = (
        data["Machine failure"]
        .value_counts()
        .sort_index()
    )

    failure_data = pd.DataFrame({
        "Status": ["No Failure", "Failure"],
        "Count": [
            failure_counts.get(0, 0),
            failure_counts.get(1, 0)
        ]
    })

    failure_percentage = data["Machine failure"].mean()

    failure_chart = (
        alt.Chart(failure_data)
        .mark_arc(innerRadius=90)
        .encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color(
                "Status:N",
                title="Machine Status"
            ),
            tooltip=[
                alt.Tooltip("Status:N", title="Machine Status"),
                alt.Tooltip("Count:Q", title="Observations", format=",")
            ]
        )
        .properties(
            height=400
        )
    )

    center_text = (
        alt.Chart(
            pd.DataFrame({
                "x": [0.5],
                "y": [0.5],
                "Label": [f"{failure_percentage:.2%}\nFailure"]
            })
        )
        .mark_text(
            fontSize=20,
            fontWeight="bold",
            lineBreak="\n",
            color="white",
            align="center",
            baseline="middle",
            dy=-12
        )
        .encode(
            x=alt.X(
                "x:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=None
            ),
            y=alt.Y(
                "y:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=None
            ),
            text="Label:N"
        )
    )

    failure_chart = failure_chart + center_text

    st.altair_chart(
        failure_chart,
        width="stretch"
    )

    st.caption(
        f"The dataset is highly imbalanced, with machine failures "
        f"representing only {failure_percentage:.2%} of the "
        f"{len(data):,} observations."
    )

def display_torque_speed_scatter():
    """Display the relationship between rotational speed and torque."""

    st.subheader("Rotational Speed vs. Torque")

    scatter_data = data[
        [
            "Type",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Machine failure"
        ]
    ].copy()

    # Use simpler column names for Altair
    scatter_data = scatter_data.rename(columns={
        "Rotational speed [rpm]": "Rotational Speed",
        "Torque [Nm]": "Torque",
        "Tool wear [min]": "Tool Wear"
    })

    scatter_data["Failure Status"] = (
        scatter_data["Machine failure"]
        .map({
            0: "No Failure",
            1: "Failure"
        })
    )

    scatter_chart = (
        alt.Chart(scatter_data)
        .mark_circle(
            size=60,
            opacity=0.6
        )
        .encode(
            x=alt.X(
                "Rotational Speed:Q",
                title="Rotational Speed [rpm]"
            ),
            y=alt.Y(
                "Torque:Q",
                title="Torque [Nm]"
            ),
            color=alt.Color(
                "Failure Status:N",
                title="Machine Status"
            ),
            tooltip=[
                alt.Tooltip("Type:N", title="Product Type"),
                alt.Tooltip(
                    "Rotational Speed:Q",
                    title="Rotational Speed"
                ),
                alt.Tooltip(
                    "Torque:Q",
                    title="Torque"
                ),
                alt.Tooltip(
                    "Tool Wear:Q",
                    title="Tool Wear"
                ),
                alt.Tooltip(
                    "Failure Status:N",
                    title="Machine Status"
                )
            ]
        )
        .properties(height=400)
    )

    st.altair_chart(
        scatter_chart,
        width="stretch"
    )

    st.caption(
        "This visualization shows the relationship between rotational "
        "speed and torque for failure and non-failure observations. "
        "Failures appear more frequently in operating regions with unusually "
        "high torque or unusually high rotational speed, while substantial "
        "overlap remains between the two classes."
    )

def display_feature_importance():
    """Display Random Forest feature importance values."""

    st.subheader("Model Feature Importance")

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": classifier.feature_importances_
    })

    feature_importance["Feature"] = (
        feature_importance["Feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__Type_", "Product Type: ", regex=False)
    )

    feature_importance = feature_importance.sort_values(
        "Importance",
        ascending=False
    )

    feature_chart = (
        alt.Chart(feature_importance)
        .mark_bar()
        .encode(
            x=alt.X(
                "Importance:Q",
                title="Importance"
            ),
            y=alt.Y(
                "Feature:N",
                sort="-x",
                title="Feature",
                axis=alt.Axis(labelLimit=250)
            ),
            tooltip=[
                alt.Tooltip("Feature:N", title="Feature"),
                alt.Tooltip(
                    "Importance:Q",
                    title="Importance",
                    format=".3f"
                )
            ]
        )
        .properties(height=300)
    )

    st.altair_chart(
        feature_chart,
        width="stretch"
    )

    st.caption(
        "Feature importance represents the relative contribution of each "
        "input feature to the Random Forest model's predictions. Higher "
        "importance indicates greater influence on the model, but does not "
        "imply that the feature causes machine failure."
    )

def display_model_information():
    """Display model configuration and maintenance information."""

    with st.expander("Model Information & Maintenance"):
        st.write(
            "This application uses a Random Forest classification model "
            "trained on the AI4I 2020 Predictive Maintenance Dataset."
        )

        st.markdown(
            f"""
            **Model:** Random Forest Classifier  
            **Decision threshold:** {FAILURE_THRESHOLD:.0%}  
            **Training dataset:** AI4I 2020 Predictive Maintenance Dataset  
            **Dataset observations:** {len(data):,}  
            **Input features:** Product type, air temperature, process
            temperature, rotational speed, torque, and tool wear
            """
        )

        st.markdown("#### Model Performance")

        st.markdown(
            """
            **Precision:** 61.2%  
            **Recall:** 76.5%  
            **F1 Score:** 68.0%  
            **ROC-AUC:** 0.963
            """
        )

        st.caption(
            "Performance metrics were calculated using the held-out test dataset "
            "with a 40% classification threshold."
        )

        st.markdown("#### Maintenance Guidance")

        st.write(
            "Model performance should be periodically evaluated using "
            "new equipment observations and known failure outcomes. "
            "Changes in operating conditions, equipment, or data "
            "distributions may reduce model performance over time."
        )

        st.write(
            "If performance declines, the model should be retrained using "
            "updated representative data and reevaluated using metrics such "
            "as precision, recall, F1 score, and the confusion matrix before "
            "the updated model is deployed."
        )

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

st.title("Predictive Maintenance System")

st.write(
    "This application uses a Random Forest machine learning model "
    "to estimate the potential for equipment failure based on current "
    "operating measurements."
)

display_prediction_form()

st.divider()

st.header("Dataset and Model Visualizations")

display_failure_distribution()

display_torque_speed_scatter()

display_feature_importance()

st.divider()

display_model_information()
