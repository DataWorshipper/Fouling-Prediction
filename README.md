# Predictive Modeling of Fouling Resistance in Heat Exchangers

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fouling-prediction-f9voxwrzp9teuyn65q93wp.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A data-driven **Predictive Maintenance (PdM)** framework for modeling the non-linear, discontinuous degradation of industrial shell-and-tube heat exchangers. This project transcends classical statistical baselines by engineering physics-informed temporal features to accurately predict Fouling Resistance ($R_f$) using Gradient Boosting and Temporal Fusion Transformers.

---

## Overview

Fouling in heat exchangers acts as an insulating blanket, introducing an additional thermal resistance ($R_f$) that progressively degrades the Overall Heat Transfer Coefficient ($U$). This leads to severe energy inefficiency and unplanned downtime. 

Because cleaning events (Cleaning-In-Place / CIP) periodically reset the fouling resistance to zero, the degradation follows a non-stationary **"sawtooth" pattern**. This project utilizes over **700,000 hourly operational observations** to model these kinetics, moving from reactive maintenance to an AI-driven predictive schedule.

### Key Achievements
* **Engineered Temporal Coordinates:** Designed the `hours_since_clean` feature to solve the phase-shift errors inherent in classical time-series models (ARIMA/SARIMA).
* **Champion Model (XGBoost):** Achieved an **$R^2$ of 0.9995** and MAE of 0.0041 by mastering the sharp discontinuities of CIP resets.
* **Deep Learning (TFT):** Implemented Google's Temporal Fusion Transformer for probabilistic multi-horizon forecasting and variable selection interpretability ($R^2 = 0.968$).

---

## 🌐 Live Streamlit Application

You can interact with the deployed model directly via our Digital Twin dashboard:

 **[Launch the Fouling Prediction Dashboard](https://fouling-prediction-f9voxwrzp9teuyn65q93wp.streamlit.app/)**

### How to use the Live App:
1. **Upload Telemetry:** Upload a CSV containing real-time operational sensors (e.g., `flow_actual`, `T_in_actual_K`, `is_cleaning`).
2. **Automated ETL:** The app automatically standardizes the physical inputs (Z-score scaling) and calculates the necessary temporal features (`hours_since_clean`, lags).
3. **Real-Time Prediction:** View the instantaneous predicted Fouling Resistance ($R_f$) plotted against your system's critical failure threshold.
4. **What-If Analysis:** Use the interactive sliders to adjust mass flow rates or inlet temperatures to simulate how altering operational parameters extends or reduces the Time-To-Failure (TTF).

---

## Repository Structure

The codebase follows a modular, MLOps-friendly architecture:

```text
Fouling-Prediction/
│
├── app/                  # Streamlit web application source code
├── artifacts/            # Serialized models (.ubj/.joblib) and fitted scalers (.pkl)
├── config/               # Centralized hyperparameter and path configurations
├── data/                 # Data manifests (Raw data downloaded from Kaggle)
├── etl/                  # Polars-based Extract, Transform, Load pipeline modules
├── images/               # EDA plots and architectural diagrams
├── model/                # Training scripts (xgboost, catboost, tft, lstm)
├── notebooks/            # Comprehensive Jupyter notebooks (Research & EDA)
├── requirements.txt      # Python dependencies
└── README.md## ⚙️ Prerequisites & Installation

Clone the repository and install the required dependencies. We recommend using a virtual environment.

```bash
git clone https://github.com/DataWorshipper/Fouling-Prediction.git
cd Fouling-Prediction
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the ETL Pipeline

The ETL pipeline uses **Polars** for ultra-fast, multi-threaded data processing. It handles schema standardization, Z-score scaling (saving the scaler to `/artifacts`), and temporal feature engineering.

```bash
# Process raw data and generate the modeling dataset
python etl/preprocess.py --input data/raw_data.csv --output data/processed_data.csv
```

---

## Training Models

You can train specific architectures using the scripts in the `/model` directory. Hyperparameter optimization is handled automatically via **Optuna**.

```bash
# Train the XGBoost champion model
python model/train_xgboost.py --config config/model_config.yaml

# Train the Temporal Fusion Transformer (requires GPU for reasonable training times)
python model/train_tft.py --epochs 30 --batch_size 64
```

Trained models are automatically serialized and saved to the `/artifacts` directory.

---

## Running Inference (API / Script)

To generate predictions on new unseen data using the saved artifacts:

```bash
python model/inference.py --input data/new_sensor_readings.csv --output data/predictions.csv
```

---

## Running the Streamlit App Locally

If you wish to test or modify the Streamlit dashboard locally:

```bash
streamlit run app/main.py
```

This will spin up a local web server (usually at http://localhost:8501).


