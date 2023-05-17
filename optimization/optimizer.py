import importlib
import logging
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import optuna

import constants
import data_landing
import framework_settings
import experiment_settings
from data_preprocessing.data_preprocessing import DataPreprocessing
from eda import eda
from feature_engineering.feature_engineering import FeatureEngineering
from modeling.modeling import Modeling
from modeling import modeling_constants

pd.set_option('display.expand_frame_repr', True)
pd.set_option('display.max_columns', None)  # Set to None for unlimited number of output rows.
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_info_columns', 500)
pd.set_option('display.max_rows', None)  # Set to None for unlimited number of output rows.
pd.set_option('display.width', 500)  # Width of the display in characters.

experiment_constants = importlib.import_module(
    f"experiment_configs.{experiment_settings.EXPERIMENT_CONSTANTS_MODULE_NAME}")


DEFAULT_IMPUTERS = [framework_settings.DEFAULT_IMPUTER]

# begin experiment
start = datetime.now()
log = logging.getLogger(__name__)

log.info(f"Backtesting started at: {str(start)}")


class Optimizer():
    def __init__(self, experiment_id=None):
        if experiment_id is None:
            self.experiment_id = uuid.uuid4()
        self.experiment_id = experiment_id
        self.data_loading_function = experiment_constants.DATA_LOADING_FUNCTION
        self.data = self.get_data()

        try:
            imputers = experiment_constants.IMPUTERS
            if imputers is None:
                imputers = DEFAULT_IMPUTERS
        except:
            imputers = DEFAULT_IMPUTERS
        self.imputers = imputers

        try:
            outlier_methods = experiment_constants.OUTLIER_METHODS
            if outlier_methods is None:
                outlier_methods = constants.OUTLIER_METHODS
            else:
                outlier_methods = [outlier_method for outlier_method in outlier_methods if outlier_method in constants.OUTLIER_METHODS]
        except:
            outlier_methods = constants.OUTLIER_METHODS
        self.outlier_methods = outlier_methods

        try:
            feature_engineering_scalers = experiment_constants.FEATURE_ENGINEERING_SCALERS
            if feature_engineering_scalers is None:
                feature_engineering_scalers = [framework_settings.DEFAULT_FEATURE_ENGINEERING_SCALER]
        except:
            feature_engineering_scalers = [framework_settings.DEFAULT_FEATURE_ENGINEERING_SCALER]
        self.feature_engineering_scalers = feature_engineering_scalers

        try:
            models_to_train = experiment_constants.MODELS_TO_TRAIN
            if models_to_train is None:
                models_to_train = modeling_constants.MODEL_NAMES
            elif models_to_train == 'all':
                models_to_train = modeling_constants.MODEL_NAMES
            elif isinstance(models_to_train, (list, tuple)):
                for model in models_to_train:
                    if model not in modeling_constants.MODEL_NAMES:
                        raise Exception(
                            f"invalid model name {model}")  # to be changed while adding exception handling properly
            else:
                raise Exception("invalid model name(s)")  # to be changed while adding exception handling properly
        except:
            models_to_train = modeling_constants.MODEL_NAMES
        self.models_to_train = models_to_train

        try:
            self.dataset_for_performance_optimization = experiment_constants.DATASET_FOR_PERFORMANCE_OPTIMIZATION
        except:
            self.dataset_for_performance_optimization = framework_settings.DEFAULT_DATASET_FOR_PERFORMANCE_OPTIMIZATION

    def get_data(self):
        data = data_landing.execute_data_landing(experiment_id=self.experiment_id, data_loading_function=self.data_loading_function)

        try:
            generate_eda_plots = experiment_constants.GENERATE_EDA_PLOTS
        except:
            generate_eda_plots = False

        # data, mv, mv_high, is_stationary, data_norm, data_ol = eda.execute_eda(experiment_id=self.experiment_id, data=data, plots=generate_eda_plots)
        data, mv, mv_high, is_stationary = eda.execute_eda(experiment_id=self.experiment_id, data=data, plots=generate_eda_plots)
        return data

    def run_pipeline(
            self,
            trial,
            imputer,
            outlier_method,
            feature_engineering_scaler,
            oh_encoder_min_frequency,
            model_to_train,
    ):
        train, val, test = DataPreprocessing(
            experiment_id=self.experiment_id,
            trial=trial,
            data=self.data,
            imputer=imputer,
        ).execute_preprocessing()

        categorical_feature_encoder = OneHotEncoder(handle_unknown='ignore', min_frequency=oh_encoder_min_frequency)
        train, val, test = FeatureEngineering(
            experiment_id=self.experiment_id,
            trial=trial,
            data=(train, val, test),
            outlier_method=outlier_method,
            categorical_feature_encoder=categorical_feature_encoder,
            feature_engineering_scaler=feature_engineering_scaler,
        ).execute_feature_engineering()

        model_performance = Modeling(
            experiment_id=self.experiment_id,
            trial=trial,
            data=(train, val, test),
            model=model_to_train,
        ).execute_modeling()

        return model_performance

    def get_optimization_objective(self):
        def objective(trial):
            imputer = trial.suggest_categorical("imputer", self.imputers)
            outlier_method = trial.suggest_categorical("outlier_method", self.outlier_methods)
            feature_engineering_scaler = trial.suggest_categorical("feature_engineering_scaler", self.feature_engineering_scalers)
            oh_encoder_min_frequency = trial.suggest_float("oh_encoder_min_frequency", 0, 0.5)
            model_to_train = trial.suggest_categorical("model_to_train", self.models_to_train)

            result = self.run_pipeline(
                trial,
                imputer,
                outlier_method,
                feature_engineering_scaler,
                oh_encoder_min_frequency,
                model_to_train,
            )
            result = result[self.dataset_for_performance_optimization]
            return result

        return objective

    def optimize(self, direction='minimize', n_trials=1000):
        study = optuna.create_study(direction=direction)
        objective = self.get_optimization_objective()
        study.optimize(objective, n_trials=n_trials)
        return study

    def execute_optimization(self, direction='minimize', n_trials=1000):
        study = self.optimize(direction=direction, n_trials=n_trials)
        best_study_params = study.best_params
        best_trial = study.best_trial
        imputer = best_study_params["imputer"]
        outlier_method = best_study_params["outlier_method"]
        feature_engineering_scaler = best_study_params["feature_engineering_scaler"]
        oh_encoder_min_frequency = best_study_params["oh_encoder_min_frequency"]
        model_to_train = best_study_params["model_to_train"]

        result = self.run_pipeline(
            best_trial,
            imputer,
            outlier_method,
            feature_engineering_scaler,
            oh_encoder_min_frequency,
            model_to_train,
        )
        best_study_params["model_to_train"] = (best_study_params["model_to_train"], result)
        return best_study_params
