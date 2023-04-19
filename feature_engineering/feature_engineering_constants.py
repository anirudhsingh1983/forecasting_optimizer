import importlib

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

import experiment_settings

experiment_constants = importlib.import_module(
    f"experiment_configs.{experiment_settings.EXPERIMENT_CONSTANTS_MODULE_NAME}")

if experiment_constants.FEATURE_ENGINEERING_FUNCTION is None:
    FEATURE_ENGINEERING_FUNCTION = lambda x: x
else:
    FEATURE_ENGINEERING_FUNCTION = experiment_constants.FEATURE_ENGINEERING_FUNCTION

TRAIN_VAL_SPLIT_DATE = experiment_constants.TRAIN_VAL_SPLIT_DATE
VAL_TEST_SPLIT_DATE = experiment_constants.VAL_TEST_SPLIT_DATE

if experiment_constants.CATEGORICAL_ENGINEERED_FEATURES is None:
    CATEGORICAL_ENGINEERED_FEATURES = []
else:
    CATEGORICAL_ENGINEERED_FEATURES = experiment_constants.CATEGORICAL_ENGINEERED_FEATURES

if experiment_constants.CATEGORICAL_FEATURE_ENCODER is None:
    CATEGORICAL_FEATURE_ENCODER = OneHotEncoder(handle_unknown='ignore', min_frequency=0.02)
else:
    CATEGORICAL_FEATURE_ENCODER = experiment_constants.CATEGORICAL_FEATURE_ENCODER

if experiment_constants.FEATURE_ENGINEERING_SCALER is None:
    FEATURE_ENGINEERING_SCALER = MinMaxScaler()
else:
    FEATURE_ENGINEERING_SCALER = experiment_constants.FEATURE_ENGINEERING_SCALER
