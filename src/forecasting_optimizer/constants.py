# Copyright 2023 Anirudh Singh
# SPDX-License-Identifier: BSD-3-Clause

"""Shared names used throughout the forecasting optimizer pipeline."""

# Standard package strings.
OBJECT_NAME = "object"

# Framework artifact names.
OUTPUT_DATA_FOLDER = "./output_data"
LANDING_DATA_NAME = "landed_data"
EDA_NAME = "eda"
DATA_PREPROCESSING_NAME = "processed_data"
FEATURE_ENGINEERING_NAME = "feature_engineering"
MODELING_NAME = "modeling"
TARGET_NAME = "target"
PREDICTION_NAME = "prediction"
BEST_PARAMETERS_NAME = "best_params"

# Dataset split names.
TRAIN_NAME = "train"
VAL_NAME = "val"
TEST_NAME = "test"

SLIDING_WINDOW_NAME = "sliding"
EXPANDING_WINDOW_NAME = "expanding"

LINEARREGRESSION_NAME = "linear_regression"
RIDGE_NAME = "ridge"
LASSO_NAME = "lasso"
ELASTICNET_NAME = "elasticnet"
SARIMAX_NAME = "sarimax"
PROPHET_NAME = "prophet"
NEURAL_PROPHET_NAME = "neural_prophet"
XGB_NAME = "xgb"
LGBM_NAME = "lgbm"
CNN_NAME = "cnn"
GRU_NAME = "gru"
LSTM_NAME = "lstm"
TRANSFORMER_NAME = "transformer"
BERT_NAME = "bert"
H2O_NAME = "h2o"

MSE_NAME = "neg_mean_squared_error"
MAE_NAME = "neg_mean_absolute_error"
WAPE_NAME = "wape"
MAPE_NAME = "neg_mean_absolute_percentage_error"

TRAIN_CV_SCORE_NAME = "train_cv_score"
MODEL_PERFORMANCE_RESULT_NAME = "model_performances"

OUTLIER_METHOD_BOXPLOT_NAME = "boxplot"
OUTLIER_METHOD_ZSCORE_NAME = "zscore"
OUTLIER_METHOD_SVMONECLASS_NAME = "svmoneclass"
OUTLIER_METHOD_ISOLATIONFOREST_NAME = "isolationforest"

OUTLIER_METHODS = [
    None,
    OUTLIER_METHOD_BOXPLOT_NAME,
    OUTLIER_METHOD_ZSCORE_NAME,
    OUTLIER_METHOD_SVMONECLASS_NAME,
    OUTLIER_METHOD_ISOLATIONFOREST_NAME,
]

# Utility-function constants.
ACTUAL_DISTRIBUTION_STR = "actual_dist"

OPTUNA_MINIMIZE_DIRECTION = "minimize"
OPTUNA_MAXIMIZE_DIRECTION = "maximize"
