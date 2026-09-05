# Copyright 2023 Anirudh Singh
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the supplier-level forecasting experiment."""

import sys
import uuid

import pandas as pd
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder

EXPERIMENT_ID = uuid.uuid4()

TARGET_COL = "orders"
DATE_COL = "date"
PREPROCESSING_FUNCTION = None
TRAIN_VAL_SPLIT_DATE = "2022-01-01"
VAL_TEST_SPLIT_DATE = "2022-04-01"
CATEGORICAL_ENGINEERED_FEATURES = ["weekday", "month", "year"]
MODELS_TO_TRAIN = ["xgb"]
EVALUATION_METRIC_TO_USE = None
PREDICTION_NAME = "prediction"


def load_orders():
    """Load historical supplier-level orders for the configured countries.

    The supplier-level forecasting package lives outside this repository, so
    its project path is added at load time when necessary.

    Returns:
        A single-column DataFrame whose column is named ``orders``.
    """
    path = "/Users/as918d/PycharmProjects/jonathan_supplier_level_forecast_EDA"
    if path not in sys.path:
        sys.path.append(
            "/Users/as918d/PycharmProjects/jonathan_supplier_level_forecast_EDA"
        )

    from slf.data.orders import load_historical_orders_pivot

    country = [1.0, 2.0]
    orders = load_historical_orders_pivot(country=country)
    orders = orders.iloc[:, 0].to_frame()
    orders.columns = ["orders"]
    return orders


DATA_LOADING_FUNCTION = load_orders

MISSING_VALUE_THRESHOLD = None

# Recognized outlier names; only Isolation Forest currently removes rows:
# - 'boxplot'
# - 'zscore'
# - 'svmoneclass'
# - 'isolationforest'
OUTLIER_METHOD = "isolationforest"
REMOVE_OUTLIERS = False


def fe(df):
    """Add calendar and one-day lag features to an orders DataFrame.

    Args:
        df: Orders indexed by date. The index and feature columns are updated
            before the first, lag-less row is removed.

    Returns:
        The feature-enriched DataFrame without its first row.
    """
    df.index = pd.to_datetime(df.index)
    df["weekday"] = df.index.dayofweek
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["yesterday"] = df["orders"].shift(1)
    df = df.iloc[1:, :]
    return df


FEATURE_ENGINEERING_FUNCTION = fe

IMPUTER = IterativeImputer(max_iter=10, random_state=0)

CATEGORICAL_FEATURE_ENCODER = OneHotEncoder(
    handle_unknown="ignore", min_frequency=0.02
)
FEATURE_ENGINEERING_SCALER = None

TIMESERIES_CV = True
TIMESERIES_CV_APPROACH = None

TIMESERIES_CV_WINDOW = None
TIMESERIES_CV_STEP = None

FORECASTING_HORIZON = [1, 2, 3]

USE_SKTIME = False
