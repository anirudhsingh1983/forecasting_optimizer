import importlib

import constants
import experiment_settings

experiment_constants = importlib.import_module(
    f"experiment_configs.{experiment_settings.EXPERIMENT_CONSTANTS_MODULE_NAME}")

# models
LINEARREGRESSION_NAME = constants.LINEARREGRESSION_NAME
ELASTICNET_NAME = constants.ELASTICNET_NAME
SARIMAX_NAME = constants.SARIMAX_NAME
PROPHET_NAME = constants.PROPHET_NAME
NEURAL_PROPHET_NAME = constants.NEURAL_PROPHET_NAME
RIDGE_NAME = constants.RIDGE_NAME
LASSO_NAME = constants.LASSO_NAME
XGB_NAME = constants.XGB_NAME
LGBM_NAME = constants.LGBM_NAME
CNN_NAME = constants.CNN_NAME
GRU_NAME = constants.GRU_NAME
LSTM_NAME = constants.LSTM_NAME
TRANSFORMER_NAME = constants.TRANSFORMER_NAME
BERT_NAME = constants.BERT_NAME

MODEL_NAMES = [
    LINEARREGRESSION_NAME,
    ELASTICNET_NAME,
    SARIMAX_NAME,
    PROPHET_NAME,
    NEURAL_PROPHET_NAME,
    RIDGE_NAME,
    LASSO_NAME,
    XGB_NAME,
    LGBM_NAME,
    CNN_NAME,
    GRU_NAME,
    LSTM_NAME,
    TRANSFORMER_NAME,
    BERT_NAME,
]

# split dates
TRAIN_VAL_SPLIT_DATE = experiment_constants.TRAIN_VAL_SPLIT_DATE
VAL_TEST_SPLIT_DATE = experiment_constants.VAL_TEST_SPLIT_DATE

# evaluation
MSE_NAME = constants.MSE_NAME
MAE_NAME = constants.MAE_NAME
WAPE_NAME = constants.WAPE_NAME
MAPE_NAME = constants.MAPE_NAME

EVALUATION_METRIC_NAMES = [
    MSE_NAME,
    MAE_NAME,
    WAPE_NAME,
    MAPE_NAME,
]

TIMESERIES_CV_APPROACHES = [
    constants.SLIDING_WINDOW_NAME,
    constants.EXPANDING_WINDOW_NAME,
]
