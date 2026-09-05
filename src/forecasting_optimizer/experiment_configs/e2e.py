# Copyright 2023 Anirudh Singh
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the end-to-end logistics forecasting experiment."""

import sys
import uuid
from pathlib import Path
from datetime import datetime
import logging
import pickle
import datetime as dt

from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# import sktime
# from sktime.utils.plotting import plot_series
# from sktime.forecasting.base import ForecastingHorizon
# from sktime.forecasting.naive import NaiveForecaster
# from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder

from forecasting_optimizer import constants
from forecasting_optimizer.projects.e2e import data
from forecasting_optimizer.util import utility_functions as uf

EXPERIMENT_ID = uuid.uuid4()

TARGET_COL = "units"
DATE_COL = "date"
PREPROCESSING_FUNCTION = None
TRAIN_VAL_SPLIT_DATE = "2022-10-01"
VAL_TEST_SPLIT_DATE = "2023-01-01"
CATEGORICAL_ENGINEERED_FEATURES = ["day", "month"]
MODELS_TO_TRAIN = [
    constants.RIDGE_NAME,
    constants.LASSO_NAME,
    constants.ELASTICNET_NAME,
    constants.LINEARREGRESSION_NAME,
    constants.XGB_NAME,
    constants.LSTM_NAME,
    constants.GRU_NAME,
    # constants.SARIMAX_NAME
]
# The focused CNN run intentionally supersedes the broader candidate set above.
MODELS_TO_TRAIN = [constants.CNN_NAME]
EVALUATION_METRIC_TO_USE = None
PREDICTION_NAME = "prediction"

DATA_LOADING_FUNCTION = data.get_data

MISSING_VALUE_THRESHOLD = None

TIMESERIES_INTERVAL_VALUE = 1
TIMESERIES_INTERVAL_UNIT = "day"
TIMESERIES_TARGET_FILL_GAP_VALUE = 0

GENERATE_EDA_PLOTS = True

# Recognized outlier names; only Isolation Forest currently removes rows:
# - 'boxplot'
# - 'zscore'
# - 'svmoneclass'
# - 'isolationforest'
OUTLIER_METHODS = ["isolationforest"]
REMOVE_OUTLIERS = False


FEATURE_ENGINEERING_FUNCTION = uf.get_date_features

IMPUTERS = [IterativeImputer(max_iter=10, random_state=0)]

CATEGORICAL_FEATURE_ENCODER = OneHotEncoder(
    handle_unknown="ignore", min_frequency=0.02
)
FEATURE_ENGINEERING_SCALER = None

CV_FOLDS = 5
TIMESERIES_CV = True
TIMESERIES_CV_APPROACH = None

# TIMESERIES_CV_WINDOW = None
# TIMESERIES_CV_STEP = None
# FORECASTING_HORIZON = [1,2,3]

USE_SKTIME = False

MODEL_PARAM_GRIDS = {
    constants.XGB_NAME: {
        "min_child_weight": [2],
        "subsample": [0.7, 1.0],
        "max_depth": [3, 7],
    },
    constants.SARIMAX_NAME: {
        "max_p": [5],
        "max_d": [3],
        "max_q": [5],
        "max_P": [5],
        "max_D": [3],
        "max_Q": [5],
    },
    constants.LSTM_NAME: {
        "units": [1],
        "return_sequences": [False],
        "activation": ["tanh"],
        "dropout": list(np.arange(0.1, 0.2, 0.1)),
    },
}

SEQ_DATA_LEN = 6
SEQ_STRIDE = 1
SEQ_SAMPLING = 1

TUNE_USING_OPTUNA = True
NUM_OPTUNA_TRIALS = 100
CNN_EPOCHS = 50
GRU_EPOCHS = 20
LSTM_EPOCHS = 20
MAX_SEQUENTIAL_NN_LAYERS = 3
MAX_CNN_LAYERS = 3
MAX_CNN_FILTERS = 5

# Supported values are ``constants.VAL_NAME`` and ``constants.TEST_NAME``.
DATASET_FOR_PERFORMANCE_OPTIMIZATION = constants.VAL_NAME
