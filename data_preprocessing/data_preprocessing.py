import importlib
import logging
from datetime import datetime

import pandas as pd

import constants
import experiment_settings
from data_preprocessing import data_preprocessing_constants
from util import utility_functions as uf

experiment_constants = importlib.import_module(
    f"experiment_configs.{experiment_settings.EXPERIMENT_CONSTANTS_MODULE_NAME}")

start = datetime.now()
log = logging.getLogger(__name__)

_preprocesing_function = data_preprocessing_constants.PREPROCESSING_FUNCTION
_train_val_split_date = data_preprocessing_constants.TRAIN_VAL_SPLIT_DATE
_val_test_split_date = data_preprocessing_constants.VAL_TEST_SPLIT_DATE


def _load_data(experiment_id):
    data = uf.read_df(experiment_id, name=constants.EDA_NAME)
    return data


def _save_processed_data(data, experiment_id):
    train, val, test = data
    uf.save_df(train, experiment_id, name=f"{constants.DATA_PREPROCESSING_NAME}_{constants.TRAIN_NAME}")
    uf.save_df(val, experiment_id, name=f"{constants.DATA_PREPROCESSING_NAME}_{constants.VAL_NAME}")
    uf.save_df(test, experiment_id, name=f"{constants.DATA_PREPROCESSING_NAME}_{constants.TEST_NAME}")


def _split_data(data, experiment_id):
    data.index = pd.to_datetime(data.index)
    train = data[data.index < _train_val_split_date]
    val = data[(data.index >= _train_val_split_date) & (data.index < _val_test_split_date)]
    test = data[data.index >= _val_test_split_date]
    _save_processed_data((train, val, test), experiment_id)
    return train, val, test


def _treat_missing_values(train, val, test, imputer):
    imputer.fit(train)
    train = pd.DataFrame(data=imputer.transform(train), index=train.index, columns=train.columns)
    val = pd.DataFrame(data=imputer.transform(val), index=val.index, columns=val.columns)
    test = pd.DataFrame(data=imputer.transform(test), index=test.index, columns=test.columns)
    return train, val, test


def execute_preprocessing(experiment_id):
    data = _load_data(experiment_id=experiment_id)
    # The _preprocesing_function should do the following:
    # General data processing (may include scaling/transformations/etc.)
    # Data filtering, as needed
    # Feature engineering pre train-val-test split
    data = _preprocesing_function(data)
    train, val, test = _split_data(data, experiment_id)
    train, val, test = _treat_missing_values(train, val, test, experiment_constants.IMPUTER)
    _save_processed_data((train, val, test), experiment_id)
