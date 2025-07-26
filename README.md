# Forecasting Optimizer

Forecasting Optimizer provides a modular pipeline to build and tune forecasting models. It integrates data landing, preprocessing, feature engineering, model training and hyper-parameter optimisation using Optuna. Example experiment configurations are located in `experiment_configs/` and project specific utilities in `projects/`.

## Installation

1. Create a virtual environment (optional but recommended).
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running an Experiment

1. Select the experiment configuration by editing `experiment_settings.py` and setting `EXPERIMENT_CONSTANTS_MODULE_NAME` to one of the modules under `experiment_configs` (e.g. `e2e`, `contact_forecasting`, `stocks`).
2. Execute the main pipeline:
   ```bash
   python main.py
   ```
3. Outputs such as intermediate datasets and model artifacts are stored in `output_data/`.

## Repository Structure

- `data_preprocessing/` – utilities for cleaning and preparing raw data.
- `feature_engineering/` – generation of features and outlier handling.
- `modeling/` – model definitions and training routines.
- `optimization/` – hyper-parameter search using Optuna.
- `projects/` – example project specific helpers and datasets.

This project requires access to Google BigQuery for some datasets and uses TensorFlow, LightGBM and XGBoost for modelling.
