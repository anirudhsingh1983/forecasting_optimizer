import importlib

import numpy as np
from sklearn.impute import SimpleImputer

import experiment_settings

experiment_constants = importlib.import_module(
    f"experiment_configs.{experiment_settings.EXPERIMENT_CONSTANTS_MODULE_NAME}")

if experiment_constants.PREPROCESSING_FUNCTION is None:
    PREPROCESSING_FUNCTION = lambda x: x
else:
    PREPROCESSING_FUNCTION = experiment_constants.PREPROCESSING_FUNCTION

TRAIN_VAL_SPLIT_DATE = experiment_constants.TRAIN_VAL_SPLIT_DATE
VAL_TEST_SPLIT_DATE = experiment_constants.VAL_TEST_SPLIT_DATE

if experiment_constants.IMPUTER is None:
    IMPUTER = SimpleImputer(missing_values=np.nan, strategy='mean')
else:
    IMPUTER = experiment_constants.IMPUTER
