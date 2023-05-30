import importlib
import itertools
import logging
import types
from abc import ABC, abstractmethod

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from lightgbm import LGBMRegressor
from pmdarima.arima import auto_arima
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.metrics import get_scorer, get_scorer_names
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_validate
from sktime.forecasting.model_selection import (
    SlidingWindowSplitter,
    ForecastingGridSearchCV,
    ExpandingWindowSplitter)
from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM, GRU, Conv2D, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Model, Sequential
from xgboost import XGBRegressor

import constants
import experiment_settings
import framework_settings
from modeling import modeling_constants
from util import utility_functions as uf

experiment_constants = importlib.import_module(
    f"experiment_configs.{experiment_settings.EXPERIMENT_CONSTANTS_MODULE_NAME}")

MODEL_NAMES = modeling_constants.MODEL_NAMES

try:
    TIMESERIES_USE_CV = experiment_constants.TIMESERIES_CV
    if TIMESERIES_USE_CV is None:
        TIMESERIES_USE_CV = framework_settings.DEFAULT_USE_TIMESERIES_CV
except:
    TIMESERIES_USE_CV = framework_settings.DEFAULT_USE_TIMESERIES_CV

try:
    TIMESERIES_CV_EQUAL_SETS = experiment_constants.TIMESERIES_CV_EQUAL_SETS
    if TIMESERIES_CV_EQUAL_SETS is None:
        TIMESERIES_CV_EQUAL_SETS = framework_settings.DEFAULT_TIMESERIES_CV_EQUAL_SETS
except:
    TIMESERIES_CV_EQUAL_SETS = framework_settings.DEFAULT_TIMESERIES_CV_EQUAL_SETS

try:
    TIMESERIES_CV_APPROACH = experiment_constants.TIMESERIES_CV_APPROACH
    if TIMESERIES_CV_APPROACH is None:
        TIMESERIES_CV_APPROACH = framework_settings.DEFAULT_TIMESERIES_CV_APPROACH
    else:
        if TIMESERIES_CV_APPROACH not in modeling_constants.TIMESERIES_CV_APPROACHES:
            raise Exception(f"Invalid timeseries CV approach {TIMESERIES_CV_APPROACH}")
except:
    TIMESERIES_CV_APPROACH = framework_settings.DEFAULT_TIMESERIES_CV_APPROACH

try:
    CV_FOLDS = experiment_constants.CV_FOLD
    if CV_FOLDS is None:
        CV_FOLDS = framework_settings.DEFAULT_CV_FOLDS
except:
    CV_FOLDS = framework_settings.DEFAULT_CV_FOLDS

try:
    TIMESERIES_CV_WINDOW = experiment_constants.TIMESERIES_CV_WINDOW
except:
    TIMESERIES_CV_WINDOW = None

try:
    TIMESERIES_CV_STEP = experiment_constants.TIMESERIES_CV_STEP
except:
    TIMESERIES_CV_STEP = None

try:
    FORECASTING_HORIZON = experiment_constants.FORECASTING_HORIZON
except:
    FORECASTING_HORIZON = None

try:
    USE_SKTIME = experiment_constants.USE_SKTIME
    if USE_SKTIME is None:
        USE_SKTIME = framework_settings.DEFAULT_USE_SKTIME
except:
    USE_SKTIME = framework_settings.DEFAULT_USE_SKTIME

try:
    PREDICTION_NAME = experiment_constants.PREDICTION_NAME
    if PREDICTION_NAME is None:
        PREDICTION_NAME = framework_settings.DEFAULT_PREDICTION_NAME
except:
    PREDICTION_NAME = framework_settings.DEFAULT_PREDICTION_NAME

try:
    EVALUATION_METRIC_TO_USE = experiment_constants.EVALUATION_METRIC_TO_USE
    if (EVALUATION_METRIC_TO_USE is None) | (EVALUATION_METRIC_TO_USE not in modeling_constants.EVALUATION_METRIC_NAMES):
        EVALUATION_METRIC_TO_USE = framework_settings.DEFAULT_EVALUATION_METRIC_TO_USE
except:
    EVALUATION_METRIC_TO_USE = framework_settings.DEFAULT_EVALUATION_METRIC_TO_USE

try:
    SEQ_DATA_LEN = experiment_constants.SEQ_DATA_LEN
    if SEQ_DATA_LEN is None:
        SEQ_DATA_LEN = framework_settings.DEFAULT_SEQ_DATA_LEN
except:
    SEQ_DATA_LEN = framework_settings.DEFAULT_SEQ_DATA_LEN

try:
    SEQ_STRIDE = experiment_constants.SEQ_STRIDE
    if SEQ_STRIDE is None:
        SEQ_STRIDE = framework_settings.DEFAULT_SEQ_STRIDE
except:
    SEQ_STRIDE = framework_settings.DEFAULT_SEQ_STRIDE

try:
    SEQ_SAMPLING = experiment_constants.SEQ_SAMPLING
    if SEQ_SAMPLING is None:
        SEQ_SAMPLING = framework_settings.DEFAULT_SEQ_SAMPLING
except:
    SEQ_SAMPLING = framework_settings.DEFAULT_SEQ_SAMPLING

try:
    MAX_CNN_LAYERS = experiment_constants.MAX_CNN_LAYERS
    if MAX_CNN_LAYERS is None:
        MAX_CNN_LAYERS = framework_settings.DEFAULT_MAX_CNN_LAYERS
except:
    MAX_CNN_LAYERS = framework_settings.DEFAULT_MAX_CNN_LAYERS

try:
    MAX_CNN_FILTERS = experiment_constants.MAX_CNN_LAYERS
except:
    MAX_CNN_FILTERS = None

try:
    MAX_SEQUENTIAL_NN_LAYERS = experiment_constants.MAX_SEQUENTIAL_NN_LAYERS
    if MAX_SEQUENTIAL_NN_LAYERS is None:
        MAX_SEQUENTIAL_NN_LAYERS = framework_settings.DEFAULT_MAX_SEQUENTIAL_NN_LAYERS
except:
    MAX_SEQUENTIAL_NN_LAYERS = framework_settings.DEFAULT_MAX_SEQUENTIAL_NN_LAYERS

RETRAIN_MODEL_ON_BEST_PARAMS = False # If False, the model will use the CV result on train set to be the train set performance.
try:
    RETRAIN_MODEL_ON_BEST_PARAMS = experiment_constants.RETRAIN_MODEL_ON_BEST_PARAMS
    if RETRAIN_MODEL_ON_BEST_PARAMS is None:
        RETRAIN_MODEL_ON_BEST_PARAMS = framework_settings.DEFAULT_RETRAIN_MODEL_ON_BEST_PARAMS
except:
    RETRAIN_MODEL_ON_BEST_PARAMS = framework_settings.DEFAULT_RETRAIN_MODEL_ON_BEST_PARAMS


try:
    TUNE_USING_OPTUNA = experiment_constants.TUNE_USING_OPTUNA
    if TUNE_USING_OPTUNA is None:
        TUNE_USING_OPTUNA = framework_settings.DEFAULT_TUNE_USING_OPTUNA
except:
    TUNE_USING_OPTUNA = framework_settings.DEFAULT_TUNE_USING_OPTUNA

try:
    NUM_OPTUNA_TRIALS = experiment_constants.NUM_OPTUNA_TRIALS
    if NUM_OPTUNA_TRIALS is None:
        NUM_OPTUNA_TRIALS = framework_settings.DEFAULT_NUM_OPTUNA_TRIALS
except:
    NUM_OPTUNA_TRIALS = framework_settings.DEFAULT_NUM_OPTUNA_TRIALS


class MdfBaseModel(ABC):
    def __init__(self, data, target_col_name, feature_engineering_scaler, experiment_id):
        self.data = data
        self.target_col_name = target_col_name
        self.model = None
        self.metric = EVALUATION_METRIC_TO_USE
        self.scorer = self.get_scorer_function(eval_metric=self.metric)
        self.scaler = feature_engineering_scaler
        self.experiment_id = experiment_id

    def split_xy(self, data):
        x = data.drop(columns=[self.target_col_name])
        y = data[self.target_col_name]
        return x, y

    def scale_cs_data(self, train, val=None, test=None):
        if self.scaler is not None:
            train = pd.DataFrame(train)
            val = pd.DataFrame(val)
            test = pd.DataFrame(test)

            self.scaler.fit(train)
            train = pd.DataFrame(
                data=self.scaler.transform(train),
                index=train.index,
                columns=train.columns,
            )

            if val is None:
                val = np.empty(shape=[0,0,0])
            else:
                if len(val) > 0:
                    val = pd.DataFrame(
                        data=self.scaler.transform(val),
                        index=val.index,
                        columns=val.columns,
                    )

            if test is None:
                test = np.empty(shape=[0,0,0])
            else:
                if len(test) > 0:
                    test = pd.DataFrame(
                        data=self.scaler.transform(test),
                        index=test.index,
                        columns=test.columns,
                    )
        return train, val, test

    def scale_seq_data(self, train, val=None, test=None):
        if self.scaler is not None:
            shape = train.shape
            newshape = tuple([shape[0] * shape[1]]) + shape[2:]

            train_df = pd.DataFrame(
                data = np.reshape(a=train, newshape=newshape, order='C'),
            ).drop_duplicates().values
            self.scaler.fit(train_df)

            train = pd.Series(list(train))
            train = train.apply(lambda df: self.scaler.transform(df.reshape([1,1])).reshape([1]) if (df.shape==(1,)) else self.scaler.transform(df))
            train = np.array(list(train))

            if val is None:
                val = np.empty(shape=[0,0,0])
            else:
                if len(val) > 0:
                    val = pd.Series(list(val))
                    val = val.apply(lambda df: self.scaler.transform(df.reshape([1, 1])).reshape([1]) if (
                                df.shape == (1,)) else self.scaler.transform(df))
                    val = np.array(list(val))

            if test is None:
                test = np.empty(shape=[0,0,0])
            else:
                if len(test) > 0:
                    test = pd.Series(list(test))
                    test = test.apply(lambda df: self.scaler.transform(df.reshape([1, 1])).reshape([1]) if (
                                df.shape == (1,)) else self.scaler.transform(df))
                    test = np.array(list(test))

        return train, val, test

    def get_cv_attributes(self, data):
        default_timeseries_cv_window = len(data) // (CV_FOLDS + 1)
        default_timeseries_cv_step = len(data) // (CV_FOLDS + 1)
        default_forecasting_horizon = np.arange(1, default_timeseries_cv_window + 1)
        timeseries_cv_window = TIMESERIES_CV_WINDOW if TIMESERIES_CV_WINDOW is not None else default_timeseries_cv_window
        timeseries_cv_step = TIMESERIES_CV_STEP if TIMESERIES_CV_STEP is not None else default_timeseries_cv_step
        forecasting_horizon = FORECASTING_HORIZON if FORECASTING_HORIZON is not None else default_forecasting_horizon
        return forecasting_horizon, timeseries_cv_step, timeseries_cv_window

    def get_cv(self, timeseries_cv_equal_sets, data, seq_data=False):
        forecasting_horizon, timeseries_cv_step, timeseries_cv_window = self.get_cv_attributes(data)

        if seq_data:
            timeseries_cv_window = timeseries_cv_window - SEQ_DATA_LEN

        if timeseries_cv_equal_sets:
            if TIMESERIES_CV_APPROACH == constants.SLIDING_WINDOW_NAME:
                cv = TimeSeriesSplit(
                    n_splits=CV_FOLDS,
                    max_train_size=timeseries_cv_window,
                )
            elif TIMESERIES_CV_APPROACH == constants.EXPANDING_WINDOW_NAME:
                cv = TimeSeriesSplit(
                    n_splits=CV_FOLDS,
                    max_train_size=None,  # this will make the splits use expanding window
                )
            else:
                logging.error("Invalid timeseries CV approach")
        else:
            if TIMESERIES_CV_APPROACH == constants.SLIDING_WINDOW_NAME:
                cv = SlidingWindowSplitter(
                    fh=forecasting_horizon,
                    window_length=timeseries_cv_window,
                    step_length=timeseries_cv_step,
                    start_with_window=True,
                )
            elif TIMESERIES_CV_APPROACH == constants.EXPANDING_WINDOW_NAME:
                cv = ExpandingWindowSplitter(
                    fh=forecasting_horizon,
                    initial_window=timeseries_cv_window,
                    step_length=timeseries_cv_step,
                )
            else:
                logging.error("Invalid timeseries CV approach")

        cv = list(cv.split(data))
        return cv

    def save_model_outputs(self, data):
        train, val, test = data
        uf.save_df(train, self.experiment_id, name=f"{constants.MODELING_NAME}_{constants.TRAIN_NAME}")
        uf.save_df(val, self.experiment_id, name=f"{constants.MODELING_NAME}_{constants.VAL_NAME}")
        uf.save_df(test, self.experiment_id, name=f"{constants.MODELING_NAME}_{constants.TEST_NAME}")

    def get_scorer_function(self, eval_metric=EVALUATION_METRIC_TO_USE):
        if isinstance(eval_metric, str):
            if eval_metric in get_scorer_names():
                scorer = get_scorer(eval_metric)
            else:
                raise Exception(f"Scoring metric name {eval_metric} is not valid.")
        elif isinstance(eval_metric, types.FunctionType):
            scorer = eval_metric

        return scorer

    def tune_with_optuna(self, objective, direction=constants.OPTUNA_MINIMIZE_DIRECTION, n_trials=100):
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        logging.info(f"Optimized {self.metric}: {study.best_value:.5f}")
        return study

    @abstractmethod
    def get_optuna_params(self, trial, **kwargs):
        """
        abstract function to be overridden sub-class implementing concrete model fitting
        """
        pass

    def predict(self, x):
        prediction = self.model.predict(x)
        return prediction

    def evaluate(self):
        train, val, test, x_train, y_train, x_val, y_val, x_test, y_test = self.datasets
        y_train_predict = self.predict(x_train)

        if len(x_val) > 0:
            y_val_predict = self.predict(x_val)
        if len(x_test) > 0:
            y_test_predict = self.predict(x_test)

        train_targets = pd.DataFrame(
            {'actuals': np.array(y_train).squeeze(), 'predicted': np.array(y_train_predict).squeeze()})

        if len(x_val) > 0:
            val_targets = pd.DataFrame(
                {'actuals': np.array(y_val).squeeze(), 'predicted': np.array(y_val_predict).squeeze()})
        else:
            val_targets = pd.DataFrame(columns = train_targets.columns)

        if len(x_test) > 0:
            test_targets = pd.DataFrame(
                {'actuals': np.array(y_test).squeeze(), 'predicted': np.array(y_test_predict).squeeze()})
        else:
            test_targets = pd.DataFrame(columns=train_targets.columns)

        self.save_model_outputs(data=(train_targets, val_targets, test_targets))

        train_metric = self.scorer._score_func(y_train, y_train_predict)
        if len(y_val) > 0:
            val_metric = self.scorer._score_func(y_val, y_val_predict)
        else:
            val_metric = None
        if len(y_test) > 0:
            test_metric = self.scorer._score_func(y_test, y_test_predict)
        else:
            test_metric = None

        return train_metric, val_metric, test_metric


class MdfTsBaseModel(MdfBaseModel):
    def __init__(self, data, target_col_name, feature_engineering_scaler, experiment_id):
        super().__init__(data, target_col_name, feature_engineering_scaler, experiment_id)

    def get_ts_datasets(self, d=0):
        train, val, test = self.data
        val = pd.concat([train.tail(d), val], axis=0)
        test = pd.concat([val.tail(d), test], axis=0)
        x_train, y_train = self.split_xy(train)
        x_val, y_val = self.split_xy(val)
        x_test, y_test = self.split_xy(test)
        return train, val, test, x_train, y_train, x_val, y_val, x_test, y_test

    @abstractmethod
    def get_cv_score(self, X, y, cv, params):
        pass

    @abstractmethod
    def get_optuna_model_objective(estimator, X, y, cv):
        pass

    @abstractmethod
    def train_model(self, X, y, params):
        pass

    def ts_train_with_timeseries_cv(self, estimator_class, param_grid):
        data, _, _, x, y, _, _, _, _ = self.datasets
        cv = self.get_cv(timeseries_cv_equal_sets=TIMESERIES_CV_EQUAL_SETS, data=data)

        if TUNE_USING_OPTUNA:
            objective = self.get_optuna_model_objective(X=x, y=y, cv=cv)
            study = self.tune_with_optuna(objective, direction=constants.OPTUNA_MINIMIZE_DIRECTION, n_trials=NUM_OPTUNA_TRIALS)
            # estimator_obj = estimator_class(y=y, X=x, **study.best_params)
            # estimator_obj = estimator_obj.fit(y=y, X=x)
            # return estimator_obj, study.best_params
            best_params = study.best_params
            best_params_cv_score = self.get_cv_score(x, y, cv, best_params)
        else:
            param_grid = [dict(zip(list(param_grid.keys()), comb)) for comb in
                          list(itertools.product(*param_grid.values()))]
            results = []
            for params in param_grid:
                score = self.get_cv_score(self, x, y, cv, params)
                # scores = []
                # for train_index, test_index in cv:
                #     train_x = x.iloc[train_index]
                #     train_y = y.iloc[train_index]
                #     test_x = x.iloc[test_index]
                #     test_y = y.iloc[test_index]
                #     train_x, test_x = self.scale_cs_data(train=train_x, val=test_x, test=None)
                #     train_y, test_y = self.scale_cs_data(train=train_y, val=test_y, test=None)
                #     model = estimator_class(y=train_y, X=train_x, **params)
                #     model = model.fit(y=train_y, X=train_x)
                #     y_pred = model.predict(n_periods=len(test_x), X=test_x)
                #     score = self.scorer._score_func(test_y, y_pred)
                #     scores.append(score)
                # results.append((params, np.mean(scores)))
                results.append((params, score))

            best_params, best_params_cv_score = sorted(results, key=lambda x: x[1])[0]
            # estimator_obj = estimator_class(y=y, X=x, **best_result[0])
            # estimator_obj = estimator_obj.fit(y=y, X=x)
            # return estimator_obj, best_result[0], best_params_cv_score

        model = self.train_model(X=x, y=y, params=best_params)

        return  model, best_params, best_params_cv_score


class MdfCsBaseModel(MdfBaseModel):
    def __init__(self, data, target_col_name, feature_engineering_scaler, experiment_id):
        super().__init__(data, target_col_name, feature_engineering_scaler, experiment_id)

    def get_cs_datasets(self):
        train, val, test = self.data
        x_train, y_train = self.split_xy(train)
        x_val, y_val = self.split_xy(val)
        x_test, y_test = self.split_xy(test)
        return train, val, test, x_train, y_train, x_val, y_val, x_test, y_test

    def get_optuna_cs_objective(self, estimator, get_optuna_params, X, y, cv, scoring):
        def objective(trial):
            params = get_optuna_params(trial)
            model = estimator.set_params(**params)
            scores = []
            for train_index, test_index in cv:
                train_x = X.iloc[train_index]
                train_y = y.iloc[train_index]
                test_x = X.iloc[test_index]
                test_y = y.iloc[test_index]
                train_x, test_x, _ = self.scale_cs_data(train=train_x, val=test_x, test=None)
                train_y, test_y, _ = self.scale_cs_data(train=train_y, val=test_y, test=None)
                model = model.fit(X=train_x, y=train_y)
                y_pred = model.predict(X=test_x)
                score = self.scorer._score_func(test_y, y_pred)
                scores.append(score)

            mean_score = np.mean(scores)
            return mean_score

        return objective

    def cs_train_with_sklearn_cv(self, model, param_grid, data):
        gscv = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=self.scorer,
            cv=CV_FOLDS,
            refit=True,
        )
        x, y = self.split_xy(data)
        gscv.fit(X=x, y=y)
        model = model.set_params(**gscv.best_params)
        cv = KFold(n_splits=CV_FOLDS)
        scores = []
        for train_index, test_index in cv:
            train_x = x.iloc[train_index]
            train_y = y.iloc[train_index]
            test_x = x.iloc[test_index]
            test_y = y.iloc[test_index]
            train_x, test_x, _ = self.scale_cs_data(train=train_x, val=test_x, test=None)
            train_y, test_y, _ = self.scale_cs_data(train=train_y, val=test_y, test=None)
            model = model.fit(X=train_x, y=train_y)
            y_pred = model.predict(X=test_x)
            score = self.scorer._score_func(test_y, y_pred)
            scores.append(score)

        best_params_cv_score = np.mean(scores)
        model = model.fit(y=y, X=x)
        return model, gscv.best_params_, best_params_cv_score

    def cs_train_with_timeseries_cv(self, model, param_grid):
        data, _, _, x, y, _, _, _, _ = self.datasets

        if TUNE_USING_OPTUNA:
            cv = self.get_cv(timeseries_cv_equal_sets=TIMESERIES_CV_EQUAL_SETS, data=data)

            objective = self.get_optuna_cs_objective(
                estimator=model,
                get_optuna_params=self.get_optuna_params,
                X=x,
                y=y,
                cv=cv,
                scoring=self.metric,
            )
            study = self.tune_with_optuna(objective, direction=constants.OPTUNA_MINIMIZE_DIRECTION, n_trials=NUM_OPTUNA_TRIALS)
            model = model.set_params(**study.best_params)
            scores = []
            for train_index, test_index in cv:
                train_x = x.iloc[train_index]
                train_y = y.iloc[train_index]
                test_x = x.iloc[test_index]
                test_y = y.iloc[test_index]
                train_x, test_x, _ = self.scale_cs_data(train=train_x, val=test_x, test=None)
                train_y, test_y, _ = self.scale_cs_data(train=train_y, val=test_y, test=None)
                model = model.fit(X=train_x, y=train_y)
                y_pred = model.predict(X=test_x)
                score = self.scorer._score_func(test_y, y_pred)
                scores.append(score)

            best_params_cv_score = np.mean(scores)
            model = model.fit(y=y, X=x)
            return model, study.best_params, best_params_cv_score
        else:
            cv = self.get_cv(timeseries_cv_equal_sets=False, data=data)

            if USE_SKTIME:
                gscv = ForecastingGridSearchCV(
                    forecaster=model,
                    param_grid=param_grid,
                    scoring=self.scorer,
                    cv=cv,
                    n_jobs=-1,
                    refit=True,
                )
            else:
                cv = list(cv.split(data))
                gscv = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring=self.scorer,
                    cv=cv,
                    n_jobs=-1,
                    refit=True,
                )

            gscv.fit(y=y, X=x)
            model = model.set_params(**gscv.best_params_)
            scores = cross_validate(
                estimator=model,
                X=x,
                y=y,
                cv=cv,
                scoring=self.metric,
                n_jobs=-1
            )
            best_params_cv_score = -scores["test_score"].mean()
            model = model.fit(y=y, X=x)
            return model, gscv.best_params_, best_params_cv_score


class MdfSeqBaseModel(MdfBaseModel):
    def __init__(self, data, target_col_name, feature_engineering_scaler, experiment_id):
        super().__init__(data, target_col_name, feature_engineering_scaler, experiment_id)
        self.loss_mappings = {
            modeling_constants.MSE_NAME: tf.keras.metrics.mean_squared_error,
            modeling_constants.MAE_NAME: tf.keras.metrics.mean_absolute_error,
            modeling_constants.MAPE_NAME: tf.keras.metrics.mean_absolute_percentage_error,
            modeling_constants.WAPE_NAME: tf.keras.metrics.mean_squared_error, # mapping WAPE to MSE because TF doesn't have any loss function for WAPE. In future improvements, we can create a custom TF loss for WAPE.
        }
        self.loss = self.loss_mappings[self.metric]
        self.model_class = None  # abstract

    def get_seq_datasets(self):
        train, val, test = self.data
        x_train, y_train = self.split_xy(train)
        x_val, y_val = self.split_xy(val)
        x_test, y_test = self.split_xy(test)
        x_train, y_train = uf.cs_to_seq(x_train, y_train, length=SEQ_DATA_LEN, sampling_rate=SEQ_SAMPLING,
                                        stride=SEQ_STRIDE)
        if len(x_val)> 0:
            x_val, y_val = uf.cs_to_seq(x_val, y_val, length=SEQ_DATA_LEN, sampling_rate=SEQ_SAMPLING, stride=SEQ_STRIDE)
        else:
            x_val, y_val = np.empty(shape=[0,0,0]), np.empty(shape=[0])

        if len(x_test) > 0:
            x_test, y_test = uf.cs_to_seq(x_test, y_test, length=SEQ_DATA_LEN, sampling_rate=SEQ_SAMPLING,
                                          stride=SEQ_STRIDE)
        else:
            x_test, y_test = np.empty(shape=[0,0,0]), np.empty(shape=[0])

        return train, val, test, x_train, y_train, x_val, y_val, x_test, y_test

    @abstractmethod
    def get_model_object(self, trial):
        pass

    def get_optimizer_optuna_args(self, trial):
        optimizer_args = {
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1),
            "beta_1": trial.suggest_float("beta_1", 0.1, 1),
            "beta_2": trial.suggest_float("beta_2", 0.1, 1),
            "epsilon": trial.suggest_float("epsilon", 10 ** (-7), 0.1),
            "amsgrad": trial.suggest_categorical("amsgrad", [False]),
            # "weight_decay": trial.suggest_categorical("weight_decay", [None]),  # the arguments here onwards are only available in tf v2.11 onwards
            # "clipnorm": trial.suggest_categorical("clipnorm", [None]),
            # "clipvalue": trial.suggest_categorical("clipvalue", [None]),
            # "global_clipnorm": trial.suggest_categorical("global_clipnorm", [None]),
            # "use_ema": trial.suggest_categorical("use_ema", [False]),
            # "ema_momentum": trial.suggest_categorical("ema_momentum", [0.99]),
            # "ema_overwrite_frequency": trial.suggest_categorical("ema_overwrite_frequency", [None]),
            # "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
            # "max_depth": trial.suggest_int("max_depth", 3, 12),
        }
        return optimizer_args

    def get_seq_cv_score(self, cv, model_params, x_train, y_train):
        scores = []
        for train_index, test_index in cv:
            train_x = x_train[train_index]
            train_y = y_train[train_index]
            test_x = x_train[test_index]
            test_y = y_train[test_index]
            train_x, test_x = self.scale_seq_data(train=train_x, val=test_x, test=None)
            train_y, test_y = self.scale_seq_data(train=train_y, val=test_y, test=None)
            input = Input(shape=(SEQ_DATA_LEN, train_x.shape[-1]))
            output = self.model_class(**model_params)(input)
            model = Model(inputs=input, outputs=output)
            model.compile(optimizer='adam', loss=self.loss, metrics=[self.loss])
            model.fit(train_x, train_y, epochs=self.training_epochs)
            score = model.evaluate(test_x, test_y)[0]
            scores.append(score)
            del model
        tf.keras.backend.clear_session()
        return scores

    def get_optuna_seq_cv_score(self, trial, X, y, cv):
        optimizer_args = self.get_optimizer_optuna_args(trial)
        scores = []

        for train_index, test_index in cv:
            train_x = X[train_index]
            train_y = y[train_index]
            test_x = X[test_index]
            test_y = y[test_index]
            train_x, test_x, _ = self.scale_seq_data(train=train_x, val=test_x, test=None)
            train_y, test_y, _ = self.scale_seq_data(train=train_y, val=test_y, test=None)
            model = self.get_model_object(trial)
            optimizer = tf.keras.optimizers.Adam(**optimizer_args)
            model.compile(optimizer=optimizer, loss=self.loss, metrics=[self.loss])
            model.fit(train_x, train_y, epochs=self.training_epochs)
            score = model.evaluate(test_x, test_y)[0]
            scores.append(score)
            tf.keras.backend.clear_session() # clean up the space occupied by the model objects in keras session to avoid memory issues
        return np.mean(scores), optimizer

    def get_optuna_seq_objective(self, X, y, cv):
        def objective(trial):
            score, _ = self.get_optuna_seq_cv_score(trial, X, y, cv)
            return score
        return objective

    def seq_train_with_timeseries_cv(self, param_grid):
        _, _, _, x_train, y_train, x_val, y_val, _, _ = self.datasets

        cv = self.get_cv(timeseries_cv_equal_sets=TIMESERIES_CV_EQUAL_SETS, data= pd.Index(np.arange(0,len(x_train))), seq_data=True)

        if TUNE_USING_OPTUNA:
            objective = self.get_optuna_seq_objective(X=x_train, y=y_train, cv=cv)
            study = self.tune_with_optuna(objective, direction=constants.OPTUNA_MINIMIZE_DIRECTION, n_trials=NUM_OPTUNA_TRIALS)
            best_trial = study.best_trial
            best_params = study.best_params

            best_params_cv_score, optimizer = self.get_optuna_seq_cv_score(best_trial, x_train, y_train, cv)

            model = self.get_model_object(best_trial)
            model.compile(optimizer=optimizer, loss=self.loss, metrics=[self.loss])
            model.fit(y=y_train, x=x_train)
            return model, best_params, best_params_cv_score
        else:
            param_grid = [dict(zip(list(param_grid.keys()), comb)) for comb in
                          list(itertools.product(*param_grid.values()))]
            results = []
            for model_params in param_grid:
                scores = self.get_seq_cv_score(cv, model_params, x_train, y_train)
                results.append((model_params, np.mean(scores)))

            best_params = sorted(results, key=lambda x: x[1])[0][0]

            scores = self.get_seq_cv_score(cv, best_params, x_train, y_train)
            best_params_cv_score = np.mean(scores)

            input = Input(shape=(SEQ_DATA_LEN, x_train.shape[-1]))
            output = self.model_class(**best_params)(input)
            model = Model(inputs=input, outputs=output)

            model.compile(optimizer='adam', loss=self.loss, metrics=[self.loss])
            model.fit(x_train, y_train, epochs=self.training_epochs)
            return model, best_params, best_params_cv_score


class MdfSarimax(MdfTsBaseModel):
    def __init__(self, data, target_col_name, feature_engineering_scaler, experiment_id):
        super().__init__(data, target_col_name, feature_engineering_scaler, experiment_id)
        self.model_name = constants.SARIMAX_NAME
        self.default_param_grid = {
            'max_p': np.arange(2, 10, 2),
            'max_d': np.arange(1, 5, 2),
            'max_q': np.arange(2, 10, 2),
            'max_P': np.arange(2, 5, 2),
            'max_D': np.arange(1, 3),
            'max_Q': np.arange(2, 5, 2),
        }
        self.datasets = self.get_ts_datasets(d=0)

    def fit(self):
        estimator_class = auto_arima
        try:
            param_grid = experiment_constants.MODEL_PARAM_GRIDS[self.model_name]
        except:
            param_grid = self.default_param_grid

        model, best_params, best_params_cv_score = self.ts_train_with_timeseries_cv(estimator_class=estimator_class, param_grid=param_grid)

        logging.info(f"The best parameters of the {self.model_name} model are: {best_params}")

        self.model = model
        return self, best_params, best_params_cv_score

    def predict(self,
                x):  # over-riding predict of the parent class to handle specific prediction signature of auto_arima.predict.
        prediction = self.model.predict(len(x), x)
        return prediction

    def get_optuna_params(self, trial):
        params = {
            'max_p': trial.suggest_int("max_p", 2, 10, step=2),
            'max_d': trial.suggest_int("max_d", 2, 10, step=2),
            'max_q': trial.suggest_int("max_q", 2, 10, step=2),
            'max_P': trial.suggest_int("max_P", 2, 10, step=2),
            'max_D': trial.suggest_int("max_D", 2, 10, step=2),
            'max_Q': trial.suggest_int("max_Q", 2, 10, step=2),
        }
        return params

    def get_optuna_model_objective(self, X, y, cv):
        def objective(trial):
            params = self.get_optuna_params(trial)
            return self.get_cv_score(X, y, cv, params)

        return objective

    def get_cv_score(self, X, y, cv, params):
        scores = []
        for train_index, test_index in cv:
            train_x = X.iloc[train_index]
            train_y = y.iloc[train_index]
            test_x = X.iloc[test_index]
            test_y = y.iloc[test_index]
            model = auto_arima(y=train_y, X=train_x, **params)
            model = model.fit(y=train_y, X=train_x)
            y_pred = model.predict(n_periods=len(test_x), X=test_x)
            score = self.scorer._score_func(test_y, y_pred)
            scores.append(score)
        return np.mean(scores)

    def train_model(self, X, y, params):
        model = auto_arima(y=y, X=X, **params)
        model = model.fit(y=y, X=X)
        return model


class MdfProphet(MdfTsBaseModel):
    def __init__(self, data, target_col_name, feature_engineering_scaler, experiment_id):
        super().__init__(data, target_col_name, feature_engineering_scaler, experiment_id)
        self.model_name = constants.PROPHET_NAME


class MdfNeuralProphet(MdfTsBaseModel):
    def __init__(self, data, target_col_name, feature_engineering_scaler, experiment_id):
        super().__init__(data, target_col_name, feature_engineering_scaler, experiment_id)
        self.model_name = constants.NEURAL_PROPHET_NAME


class MdfLinearRegression(MdfCsBaseModel):
    def __init__(self, data, target_col_name, feature_engineering_scaler, experiment_id):
        super().__init__(data, target_col_name, feature_engineering_scaler, experiment_id)
        self.model_name = constants.LINEARREGRESSION_NAME
        self.default_param_grid = {
            'fit_intercept': [True, False],
        }
        self.datasets = self.get_cs_datasets()

    def fit(self):
        train, val, test = self.data
        model = LinearRegression()
        try:
            param_grid = experiment_constants.MODEL_PARAM_GRIDS[self.model_name]
        except:
            param_grid = self.default_param_grid

        if TIMESERIES_USE_CV:
            model, best_params, best_params_cv_score = self.cs_train_with_timeseries_cv(model=model, param_grid=param_grid)
            logging.info(f"The best parameters of the {self.model_name} model are: {best_params}")
        else:
            model, best_params, best_params_cv_score = self.cs_train_with_sklearn_cv(model=model, param_grid=param_grid, data=train)

        self.model = model
        return self, best_params, best_params_cv_score

    def get_optuna_params(self, trial):
        params = {
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        }
        return params


class MdfRidge(MdfCsBaseModel):
    def __init__(self, data, target_col_name, feature_engineering_scaler, experiment_id):
        super().__init__(data, target_col_name, feature_engineering_scaler, experiment_id)
        self.model_name = constants.RIDGE_NAME
        self.default_param_grid = {
            'alpha': [0.1, 0.5, 0.8, 1.0, 2.0],
            'fit_intercept': [True, False],
        }
        self.datasets = self.get_cs_datasets()

    def fit(self):
        train, val, test = self.data
        model = Ridge()
        try:
            param_grid = experiment_constants.MODEL_PARAM_GRIDS[self.model_name]
        except:
            param_grid = self.default_param_grid

        if TIMESERIES_USE_CV:
            model, best_params, best_params_cv_score = self.cs_train_with_timeseries_cv(model=model, param_grid=param_grid)
            logging.info(f"The best parameters of the {self.model_name} model are: {best_params}")
        else:
            model, best_params, best_params_cv_score = self.cs_train_with_sklearn_cv(model=model, param_grid=param_grid, data=train)

        self.model = model
        return self, best_params, best_params_cv_score

    def get_optuna_params(self, trial):
        params = {
            "alpha": trial.suggest_float("alpha", 0, 4, step=0.1),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        }
        return params


class MdfLasso(MdfCsBaseModel):
    def __init__(self, data, target_col_name, feature_engineering_scaler, experiment_id):
        super().__init__(data, target_col_name, feature_engineering_scaler, experiment_id)
        self.model_name = constants.LASSO_NAME
        self.default_param_grid = {
            'alpha': [0.1, 0.5, 0.8, 1.0, 2.0],
            'fit_intercept': [True, False],
        }
        self.datasets = self.get_cs_datasets()

    def fit(self):
        train, val, test = self.data
        model = Lasso()
        try:
            param_grid = experiment_constants.MODEL_PARAM_GRIDS[self.model_name]
        except:
            param_grid = self.default_param_grid

        if TIMESERIES_USE_CV:
            model, best_params, best_params_cv_score = self.cs_train_with_timeseries_cv(model=model, param_grid=param_grid)
            logging.info(f"The best parameters of the {self.model_name} model are: {best_params}")
        else:
            model, best_params, best_params_cv_score = self.cs_train_with_sklearn_cv(model=model, param_grid=param_grid, data=train)

        self.model = model
        return self, best_params, best_params_cv_score

    def get_optuna_params(self, trial):
        params = {
            "alpha": trial.suggest_float("alpha", 0, 4, step=0.1),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        }
        return params


class MdfElasticNet(MdfCsBaseModel):
    def __init__(self, data, target_col_name, feature_engineering_scaler, experiment_id):
        super().__init__(data, target_col_name, feature_engineering_scaler, experiment_id)
        self.model_name = constants.ELASTICNET_NAME
        self.default_param_grid = {
            'alpha': [0.1, 0.5, 0.8, 1.0, 2.0],
            'l1_ratio': list(np.arange(0, 1.01, 0.1)),
            'fit_intercept': [True, False],
        }
        self.datasets = self.get_cs_datasets()

    def fit(self):
        train, val, test = self.data
        model = ElasticNet()
        try:
            param_grid = experiment_constants.MODEL_PARAM_GRIDS[self.model_name]
        except:
            param_grid = self.default_param_grid

        if TIMESERIES_USE_CV:
            model, best_params, best_params_cv_score = self.cs_train_with_timeseries_cv(model=model, param_grid=param_grid)
            logging.info(f"The best parameters of the {self.model_name} model are: {best_params}")
        else:
            model, best_params, best_params_cv_score = self.cs_train_with_sklearn_cv(model=model, param_grid=param_grid, data=train)

        self.model = model
        return self, best_params, best_params_cv_score

    def get_optuna_params(self, trial):
        params = {
            "alpha": trial.suggest_float("alpha", 0.1, 4, step=0.1),
            'l1_ratio': trial.suggest_float("l1_ratio", 0, 1, step=0.05),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        }
        return params


class MdfXgb(MdfCsBaseModel):
    def __init__(self, data, target_col_name, feature_engineering_scaler, experiment_id):
        super().__init__(data, target_col_name, feature_engineering_scaler, experiment_id)
        self.model_name = constants.XGB_NAME
        self.default_param_grid = {
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5, 6, 7]
        }
        self.datasets = self.get_cs_datasets()

    def fit(self):
        train, val, test = self.data
        model = XGBRegressor()
        try:
            param_grid = experiment_constants.MODEL_PARAM_GRIDS[self.model_name]
        except:
            param_grid = self.default_param_grid

        if TIMESERIES_USE_CV:
            model, best_params, best_params_cv_score = self.cs_train_with_timeseries_cv(model=model, param_grid=param_grid)
            logging.info(f"The best parameters of the {self.model_name} model are: {best_params}")
        else:
            model, best_params, best_params_cv_score = self.cs_train_with_sklearn_cv(model=model, param_grid=param_grid, data=train)

        self.model = model
        return self, best_params, best_params_cv_score

    def get_optuna_params(self, trial):
        params = {
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 100, step=5),
            'gamma': trial.suggest_float("gamma", 0.1, 5, step=0.5),
            'subsample': trial.suggest_float("subsample", 0.1, 1, step=0.1),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.1, 1, step=0.1),
            "max_depth": trial.suggest_int("max_depth", 1, 10, step=1),
        }
        return params


class MdfLgbm(MdfCsBaseModel):
    def __init__(self, data, target_col_name, feature_engineering_scaler, experiment_id):
        super().__init__(data, target_col_name, feature_engineering_scaler, experiment_id)
        self.model_name = constants.LGBM_NAME
        self.default_param_grid = {
            "n_estimators": [10, 100, 1000, 10000, 100000],
            "learning_rate": list(np.arange(0.1, 2, 0.1)),
            "num_leaves": [20, 200, 2000, 30000],
            "max_depth": [3, 5, 7, 8, 10, 12],
            "min_data_in_leaf": [200, 1000, 5000, 10000],
            "max_bin": [200, 250, 300],
            "lambda_l1": [0, 10, 50, 100],
            "lambda_l2": [0, 10, 50, 100],
            "min_gain_to_split": [0, 1, 5, 10, 15],
            "bagging_fraction": [0.2, 0.4, 0.6, 0.8, 0.95],
            "bagging_freq": [1],
            "feature_fraction": [0.2, 0.4, 0.6, 0.8, 0.95],
        }
        self.datasets = self.get_cs_datasets()

    def fit(self):
        train, val, test = self.data
        model = LGBMRegressor()
        try:
            param_grid = experiment_constants.MODEL_PARAM_GRIDS[self.model_name]
        except:
            param_grid = self.default_param_grid

        if TIMESERIES_USE_CV:
            model, best_params, best_params_cv_score = self.cs_train_with_timeseries_cv(model=model, param_grid=param_grid)
            logging.info(f"The best parameters of the {self.model_name} model are: {best_params}")
        else:
            model, best_params, best_params_cv_score = self.cs_train_with_sklearn_cv(model=model, param_grid=param_grid, data=train)

        self.model = model
        return self, best_params, best_params_cv_score

    def get_optuna_params(self, trial):
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
            "max_bin": trial.suggest_int("max_bin", 200, 300),
            "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
            "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.2, 0.95, step=0.1
            ),
            "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.2, 0.95, step=0.1
            ),
        }
        return params


class MdfCnn(MdfSeqBaseModel):
    def __init__(self, data, target_col_name, feature_engineering_scaler, experiment_id):
        super().__init__(data, target_col_name, feature_engineering_scaler, experiment_id)
        self.model_name = None  # abstract
        self.default_param_grid = {
            "filters": np.arange(1,5),
            "kernel_size": [[3,3]],
            "activation": ['tanh', 'sigmoid', 'relu'],
            "use_bias": [True, False],
        }
        self.datasets = self.get_seq_datasets()
        self.model_name = constants.CNN_NAME

        try:
            self.training_epochs = experiment_constants.CNN_EPOCHS
            if self.training_epochs is None:
                self.training_epochs = framework_settings.DEFAULT_CNN_EPOCHS
        except:
            self.training_epochs = framework_settings.DEFAULT_CNN_EPOCHS

    def get_optuna_params(self, trial, max_kernel_height, layer_name=str(1)):
        if MAX_CNN_FILTERS is None:
            max_cnn_filters = self.datasets[3].shape[1]  # self.datasets[3] is X_train
        else:max_cnn_filters = MAX_CNN_FILTERS

        if max_cnn_filters > SEQ_DATA_LEN:
            max_cnn_filters = SEQ_DATA_LEN

        kernel_height = trial.suggest_int(f"kernel_height_{layer_name}", 1, max_kernel_height)
        params = {
            "filters": trial.suggest_int(f"filters_{layer_name}", 1, max_cnn_filters),
            "activation": trial.suggest_categorical(f"activation_{layer_name}", [None, 'tanh', 'sigmoid', 'relu']),
            "use_bias": trial.suggest_categorical(f"use_bias_{layer_name}", [True, False]),
        }
        return (params, kernel_height)

    def get_model_object(self, trial):
        num_layers = trial.suggest_int("num_layers", 1, MAX_CNN_LAYERS)
        first_layer_params, kernel_height = self.get_optuna_params(
            trial,
            max_kernel_height=self.datasets[3].shape[1],
            layer_name=str(1)
        )
        first_layer_params['kernel_size'] = [kernel_height, self.datasets[3].shape[-1]]

        input = Input(shape=(SEQ_DATA_LEN, self.datasets[3].shape[-1], 1))  # self.datasets[3] is X_train
        output = Conv2D(**first_layer_params)(input)

        for i in np.arange(2, num_layers+1):
            layer_params, kernel_height = self.get_optuna_params(trial, max_kernel_height=output.shape[1], layer_name=str(i))
            layer_params['kernel_size'] = [kernel_height, 1]
            output = Conv2D(**layer_params)(output)

        droprate = trial.suggest_float("droprate", 0, 0.8)
        output = Flatten()(output)
        output = Dropout(droprate)(output)
        output = Dense(1, activation="sigmoid")(output)

        model = Model(inputs=input, outputs=output)
        return model

    def fit(self):
        try:
            param_grid = experiment_constants.MODEL_PARAM_GRIDS[self.model_name]
        except:
            param_grid = self.default_param_grid

        self.model, best_params, best_params_cv_score = self.seq_train_with_timeseries_cv(param_grid=param_grid)
        return self, best_params, best_params_cv_score


class MdfSeqModel(MdfSeqBaseModel):
    def __init__(self, data, target_col_name, feature_engineering_scaler, experiment_id):
        super().__init__(data, target_col_name, feature_engineering_scaler, experiment_id)
        self.model_name = None  # abstract
        self.default_param_grid = {
            "units": [1],
            "return_sequences": [False],
            "activation": ['tanh', 'sigmoid', 'relu'],
            "dropout": list(np.arange(0.1, 0.6, 0.1)),
        }
        self.datasets = self.get_seq_datasets()
        self.model_class = None  # abstract

    def get_optuna_params(self, trial, layer_name=str(1), drop_units=False):
        if drop_units:
            params = {
                "activation": trial.suggest_categorical(f"activation_{layer_name}", ['tanh', 'sigmoid', 'relu']),
                "dropout": trial.suggest_float(f"dropout_{layer_name}", 0.1, 0.6, step=0.1),
            }
        else:
            params = {
                "units": trial.suggest_int(f"units_{layer_name}", 1, self.datasets[3].shape[1]), # self.datasets[3] is X_train
                "activation": trial.suggest_categorical(f"activation_{layer_name}", ['tanh', 'sigmoid', 'relu']),
                "dropout": trial.suggest_float(f"dropout_{layer_name}", 0.1, 0.6, step=0.1),
            }
        return params

    def get_model_object(self, trial):
        num_layers = trial.suggest_int("num_layers", 1, MAX_SEQUENTIAL_NN_LAYERS)
        input = Input(shape=(SEQ_DATA_LEN, self.datasets[3].shape[-1]))  # self.datasets[3] is X_train
        if num_layers == 1:
            first_layer_params = self.get_optuna_params(trial, layer_name=str(1), drop_units=True)
            output = self.model_class(units=1, return_sequences=False, **first_layer_params)(input)
        else:
            first_layer_params = self.get_optuna_params(trial, layer_name=str(1), drop_units=False)
            output = self.model_class(return_sequences=True, **first_layer_params)(input)
            for i in np.arange(2, num_layers, 1):
                middle_layer_params = self.get_optuna_params(trial, layer_name=str(i))
                output = self.model_class(return_sequences=True, **middle_layer_params)(output)
            last_layer_params = self.get_optuna_params(trial, layer_name=str(num_layers), drop_units=True)
            output = self.model_class(units=1, return_sequences=False, **last_layer_params)(output)
        model = Model(inputs=input, outputs=output)
        return model

    def fit(self):
        try:
            param_grid = experiment_constants.MODEL_PARAM_GRIDS[self.model_name]
        except:
            param_grid = self.default_param_grid

        self.model, best_params, best_params_cv_score = self.seq_train_with_timeseries_cv(param_grid=param_grid)
        return self, best_params, best_params_cv_score


class MdfGru(MdfSeqModel):
    def __init__(self, data, target_col_name, feature_engineering_scaler, experiment_id):
        super().__init__(data, target_col_name, feature_engineering_scaler, experiment_id)
        self.model_name = constants.GRU_NAME
        self.model_class = GRU
        self.default_param_grid = {
            "units": [1],
            "return_sequences": [False],
            "activation": ['tanh', 'sigmoid', 'relu'],
            "dropout": list(np.arange(0.1, 0.6, 0.1)),
        }
        try:
            self.training_epochs = experiment_constants.GRU_EPOCHS
            if self.training_epochs is None:
                self.training_epochs = framework_settings.DEFAULT_GRU_EPOCHS
        except:
            self.training_epochs = framework_settings.DEFAULT_GRU_EPOCHS

    def get_optuna_params(self, trial, layer_name=str(1), drop_units=False):
        if drop_units:
            params = {
                "activation": trial.suggest_categorical(f"activation_{layer_name}", ['tanh', 'sigmoid', 'relu']),
                "dropout": trial.suggest_float(f"dropout_{layer_name}", 0.1, 0.6, step=0.1),
            }
        else:
            params = {
                "units": trial.suggest_int(f"units_{layer_name}", 1, self.datasets[3].shape[1]), # self.datasets[3] is X_train
                "activation": trial.suggest_categorical(f"activation_{layer_name}", ['tanh', 'sigmoid', 'relu']),
                "dropout": trial.suggest_float(f"dropout_{layer_name}", 0.1, 0.6, step=0.1),
            }
        return params



class MdfLstm(MdfSeqModel):
    def __init__(self, data, target_col_name, feature_engineering_scaler, experiment_id):
        super().__init__(data, target_col_name, feature_engineering_scaler, experiment_id)
        self.model_name = constants.LSTM_NAME
        self.model_class = LSTM
        self.default_param_grid = {
            "units": [1],
            "return_sequences": [False],
            "activation": ['tanh', 'sigmoid', 'relu'],
            "dropout": list(np.arange(0.1, 0.6, 0.1)),
        }
        try:
            self.training_epochs = experiment_constants.LSTM_EPOCHS
            if self.training_epochs is None:
                self.training_epochs = framework_settings.DEFAULT_LSTM_EPOCHS
        except:
            self.training_epochs = framework_settings.DEFAULT_LSTM_EPOCHS

    def get_optuna_params(self, trial, layer_name=str(1), drop_units=False):
        if drop_units:
            params = {
                "activation": trial.suggest_categorical(f"activation_{layer_name}", ['tanh', 'sigmoid', 'relu']),
                "dropout": trial.suggest_float(f"dropout_{layer_name}", 0.1, 0.6, step=0.1),
            }
        else:
            params = {
                "units": trial.suggest_int(f"units_{layer_name}", 1, self.datasets[3].shape[1]), # self.datasets[3] is X_train
                "activation": trial.suggest_categorical(f"activation_{layer_name}", ['tanh', 'sigmoid', 'relu']),
                "dropout": trial.suggest_float(f"dropout_{layer_name}", 0.1, 0.6, step=0.1),
            }
        return params


class MdfTransformer(MdfSeqModel):
    def __init__(self, data, target_col_name, feature_engineering_scaler, experiment_id):
        super().__init__(data, target_col_name, feature_engineering_scaler, experiment_id)
        self.model_name = constants.TRANSFORMER_NAME


class MdfBert(MdfSeqModel):
    def __init__(self, data, target_col_name, feature_engineering_scaler, experiment_id):
        super().__init__(data, target_col_name, feature_engineering_scaler, experiment_id)
        self.model_name = constants.BERT_NAME


MODEL_CLASSES = [
    MdfLinearRegression,
    MdfElasticNet,
    MdfSarimax,
    MdfProphet,
    MdfNeuralProphet,
    MdfRidge,
    MdfLasso,
    MdfXgb,
    MdfLgbm,
    MdfCnn,
    MdfGru,
    MdfLstm,
    MdfTransformer,
    MdfBert,
]
MODEL_CLASSES = dict(zip(MODEL_NAMES, MODEL_CLASSES))
