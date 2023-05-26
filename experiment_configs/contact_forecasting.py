
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

import constants
from util import utility_functions as uf
from projects.contact_forecasting import support_functions as sf

EXPERIMENT_ID = uuid.uuid4()

TARGET_COL = 'presentedVolume'
DATE_COL = 'presentedDate'
PREPROCESSING_FUNCTION = sf.preprocessing
TRAIN_VAL_SPLIT_DATE = '2023-06-01'
VAL_TEST_SPLIT_DATE = '2023-06-01'
# CATEGORICAL_ENGINEERED_FEATURES = ['day', 'month']
CATEGORICAL_ENGINEERED_FEATURES = []
# MODELS_TO_TRAIN = [
#     constants.RIDGE_NAME, constants.LASSO_NAME, constants.ELASTICNET_NAME,
#     constants.LINEARREGRESSION_NAME, constants.XGB_NAME, constants.LSTM_NAME, constants.GRU_NAME, constants.CNN_NAME,
#     # constants.SARIMAX_NAME
# ]
# MODELS_TO_TRAIN = [
#     constants.ELASTICNET_NAME,
# ]
# MODELS_TO_TRAIN = [
#     constants.LSTM_NAME, constants.GRU_NAME, constants.CNN_NAME,
# ]
# MODELS_TO_TRAIN = [
#     constants.SARIMAX_NAME,
# ]
MODELS_TO_TRAIN = [
    constants.RIDGE_NAME, constants.LASSO_NAME, constants.ELASTICNET_NAME,
        constants.LINEARREGRESSION_NAME, constants.XGB_NAME
]

EVALUATION_METRIC_TO_USE = 'neg_mean_absolute_percentage_error'
PREDICTION_NAME = 'prediction'

DATA_LOADING_FUNCTION = sf.get_raw_data

MISSING_VALUE_THRESHOLD = None

TIMESERIES_INTERVAL_VALUE = 1
TIMESERIES_INTERVAL_UNIT = 'day'
TIMESERIES_TARGET_FILL_GAP_VALUE = 0

GENERATE_EDA_PLOTS = False

IMPUTING_VAL_DICT = {
    TARGET_COL: 0,
    'numOrdersTotal': 0,
    'fracNewCustomerOrders': -1,
    'usHoliday': 'None',
    'canadaHoliday': 'None',
}

# default outlier method
# Can be one of them:
# - 'boxplot'
# - 'zscore'
# - 'svmoneclass'
# - 'isolationforest'
OUTLIER_METHODS = [None]
REMOVE_OUTLIERS = False


FEATURE_ENGINEERING_FUNCTION = uf.get_date_features

IMPUTERS = [IterativeImputer(max_iter=10, random_state=0)]

# CATEGORICAL_FEATURE_ENCODER = OneHotEncoder(handle_unknown='ignore', min_frequency = 0.02)
FEATURE_ENGINEERING_SCALER = [None, MinMaxScaler()]

CV_FOLDS = 5
TIMESERIES_CV = True
TIMESERIES_CV_EQUAL_SETS = False
TIMESERIES_CV_APPROACH = None

TIMESERIES_CV_WINDOW = 190
TIMESERIES_CV_STEP = 1
FORECASTING_HORIZON = [2]

USE_SKTIME = False

MODEL_PARAM_GRIDS = {
    constants.XGB_NAME: {
        'min_child_weight': [2],
        'subsample': [0.7, 1.0],
        'max_depth': [3, 7]
        },
    constants.SARIMAX_NAME: {
            'max_p': [5],
            'max_d': [3],
            'max_q': [5],
            'max_P': [5],
            'max_D': [3],
            'max_Q': [5],
        },
    constants.LSTM_NAME: {
            "units": [1],
            "return_sequences": [False],
            "activation": ['tanh'],
            "dropout": list(np.arange(0.1, 0.2, 0.1)),
        },
}

SEQ_DATA_LEN = 6
SEQ_STRIDE = 1
SEQ_SAMPLING = 1

TUNE_USING_OPTUNA = True
NUM_OPTUNA_TRIALS = 20
CNN_EPOCHS = 50
GRU_EPOCHS = 50
LSTM_EPOCHS = 50
MAX_SEQUENTIAL_NN_LAYERS = 3
MAX_CNN_LAYERS = 3
MAX_CNN_FILTERS = 5

RETRAIN_MODEL_ON_BEST_PARAMS = False # If False, the model will use the CV result on train set to be the train set performance.
DATASET_FOR_PERFORMANCE_OPTIMIZATION = constants.TRAIN_CV_SCORE_NAME  # possible values are: [constants.TRAIN_NAME, constants.VAL_NAME, constants.TEST_NAME, constants.TRAIN_CV_SCORE_NAME]
