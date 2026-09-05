# Copyright 2023 Anirudh Singh
# SPDX-License-Identifier: BSD-3-Clause

"""Hyperparameter optimization for the complete forecasting pipeline."""

import importlib
import logging
import pickle
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import optuna
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from forecasting_optimizer import constants
from forecasting_optimizer import data_landing
from forecasting_optimizer import experiment_settings
from forecasting_optimizer import framework_settings
from forecasting_optimizer.data_preprocessing.data_preprocessing import (
    DataPreprocessing,
)
from forecasting_optimizer.eda import eda
from forecasting_optimizer.feature_engineering.feature_engineering import (
    FeatureEngineering,
)
from forecasting_optimizer.modeling import modeling_constants
from forecasting_optimizer.modeling.modeling import Modeling

# Preserve complete DataFrame output for optimization diagnostics.
pd.set_option("display.expand_frame_repr", True)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_info_columns", 500)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)

experiment_constants = importlib.import_module(
    "forecasting_optimizer.experiment_configs."
    f"{experiment_settings.EXPERIMENT_CONSTANTS_MODULE_NAME}"
)


DEFAULT_IMPUTERS = [framework_settings.DEFAULT_IMPUTER]

# Record the import-time timestamp used by the existing backtesting log.
start = datetime.now()
log = logging.getLogger(__name__)

log.info(f"Backtesting started at: {str(start)}")


class Optimizer:
    """Optimizes preprocessing, feature engineering, and model choices.

    Attributes:
        experiment_id: Identifier used for artifacts produced by the run.
        data_loading_function: Zero-argument source-data loader.
        data: EDA-prepared source dataset reused by candidate trials.
        imputers: Candidate missing-value treatments.
        outlier_methods: Candidate outlier-handling strategies.
        feature_engineering_scalers: Candidate feature scalers.
        models_to_train: Registered model names considered by optimization.
        dataset_for_performance_optimization: Metric split optimized by Optuna.
        num_optuna_processes: ``max_workers`` for the executor attached to study
            metadata; the current path does not submit trials to that executor.
    """

    def __init__(self, experiment_id=None):
        """Initializes an optimizer from the active experiment configuration.

        Args:
            experiment_id: Optional identifier for artifacts produced by this
                run.
        """
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
                outlier_methods = [
                    outlier_method
                    for outlier_method in outlier_methods
                    if outlier_method in constants.OUTLIER_METHODS
                ]
        except:
            outlier_methods = constants.OUTLIER_METHODS
        self.outlier_methods = outlier_methods

        try:
            feature_engineering_scalers = (
                experiment_constants.FEATURE_ENGINEERING_SCALERS
            )
            if feature_engineering_scalers is None:
                feature_engineering_scalers = [
                    framework_settings.DEFAULT_FEATURE_ENGINEERING_SCALER
                ]
        except:
            feature_engineering_scalers = [
                framework_settings.DEFAULT_FEATURE_ENGINEERING_SCALER
            ]
        self.feature_engineering_scalers = feature_engineering_scalers

        try:
            models_to_train = experiment_constants.MODELS_TO_TRAIN
            if models_to_train is None:
                models_to_train = modeling_constants.MODEL_NAMES
            elif models_to_train == "all":
                models_to_train = modeling_constants.MODEL_NAMES
            elif isinstance(models_to_train, (list, tuple)):
                for model in models_to_train:
                    if model not in modeling_constants.MODEL_NAMES:
                        raise Exception(f"invalid model name {model}")
            else:
                raise Exception("invalid model name(s)")
        except:
            models_to_train = modeling_constants.MODEL_NAMES
        self.models_to_train = models_to_train

        try:
            self.dataset_for_performance_optimization = (
                experiment_constants.DATASET_FOR_PERFORMANCE_OPTIMIZATION
            )
        except:
            self.dataset_for_performance_optimization = (
                framework_settings.DEFAULT_DATASET_FOR_PERFORMANCE_OPTIMIZATION
            )

        try:
            num_optuna_processes = experiment_constants.NUM_OPTUNA_PROCESSES
            if num_optuna_processes is None:
                num_optuna_processes = (
                    framework_settings.DEFAULT_NUM_OPTUNA_PROCESSES
                )
        except:
            num_optuna_processes = (
                framework_settings.DEFAULT_NUM_OPTUNA_PROCESSES
            )
        self.num_optuna_processes = num_optuna_processes

    def get_data(self):
        """Lands source data and runs exploratory analysis.

        Returns:
            The prepared dataset returned by the exploratory-analysis stage.
        """
        data = data_landing.execute_data_landing(
            experiment_id=self.experiment_id,
            data_loading_function=self.data_loading_function,
        )

        try:
            generate_eda_plots = experiment_constants.GENERATE_EDA_PLOTS
        except:
            generate_eda_plots = False

        data, mv, mv_high, is_stationary = eda.execute_eda(
            experiment_id=self.experiment_id,
            data=data,
            plots=generate_eda_plots,
        )
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
        """Runs one candidate forecasting pipeline.

        Args:
            trial: Optimization trial for the candidate run.
            imputer: Imputation strategy selected for preprocessing.
            outlier_method: Outlier-handling strategy.
            feature_engineering_scaler: Scaler passed to the feature-engineering
                object and selected model adapter. The current
                feature-engineering execution path does not apply it.
            oh_encoder_min_frequency: Minimum frequency passed to the one-hot
                encoder.
            model_to_train: Name of the forecasting model to fit.

        Returns:
            The model performance mapping produced by the modeling stage.
        """
        train, val, test = DataPreprocessing(
            experiment_id=self.experiment_id,
            trial=trial,
            data=self.data,
            imputer=imputer,
        ).execute_preprocessing()

        categorical_feature_encoder = OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=oh_encoder_min_frequency,
        )
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
            feature_engineering_scaler=feature_engineering_scaler,
            model=model_to_train,
        ).execute_modeling()

        return model_performance

    def get_optimization_objective(self):
        """Builds the Optuna objective for pipeline selection.

        Returns:
            A callable that executes a trial and returns its selected dataset
            metric.
        """

        def objective(trial):
            """Evaluates one Optuna pipeline trial.

            Args:
                trial: Optuna trial used to select pipeline parameters.

            Returns:
                The configured dataset metric for the candidate pipeline.
            """
            imputer = trial.suggest_categorical("imputer", self.imputers)
            outlier_method = trial.suggest_categorical(
                "outlier_method", self.outlier_methods
            )
            feature_engineering_scaler = trial.suggest_categorical(
                "feature_engineering_scaler",
                self.feature_engineering_scalers,
            )
            oh_encoder_min_frequency = trial.suggest_float(
                "oh_encoder_min_frequency", 0, 0.5
            )
            model_to_train = trial.suggest_categorical(
                "model_to_train", self.models_to_train
            )

            result = self.run_pipeline(
                trial,
                imputer,
                outlier_method,
                feature_engineering_scaler,
                oh_encoder_min_frequency,
                model_to_train,
            )

            try:
                with open(
                    f"tmp/optimizer_best_model_params_dict.pickle", "rb"
                ) as f:
                    optimizer_best_model_params_dict = pickle.load(f)
            except:
                optimizer_best_model_params_dict = dict()

            optimizer_best_model_params_dict[trial.number] = result
            with open(
                f"tmp/optimizer_best_model_params_dict.pickle", "wb"
            ) as f:
                pickle.dump(optimizer_best_model_params_dict, f)

            result = result[self.dataset_for_performance_optimization]
            if result is None:
                logging.error(
                    f"The error on the selected dataset is None or undefined. "
                    f"Either check if the dataset selected for optimization "
                    f"is not empty or if it is not smaller than parameter "
                    f"SEQ_DATA_LEN (or its default value DEFAULT_SEQ_DATA_LEN "
                    f"if SEQ_DATA_LEN is not defined.)"
                )
            return result

        return objective

    def optimize(self, direction="minimize", n_trials=1000):
        """Runs the configured Optuna study.

        Args:
            direction: Optimization direction accepted by Optuna.
            n_trials: Maximum number of objective trials.

        Returns:
            The completed Optuna study.
        """
        study = optuna.create_study(direction=direction)
        executor = ProcessPoolExecutor(max_workers=self.num_optuna_processes)
        study.set_user_attr("executor", executor)

        objective = self.get_optimization_objective()
        study.optimize(objective, n_trials=n_trials)
        return study

    def execute_optimization(self, direction="minimize", n_trials=1000):
        """Optimizes the pipeline and reruns its best configuration.

        Args:
            direction: Optimization direction accepted by Optuna.
            n_trials: Maximum number of objective trials.

        Returns:
            The best study parameters, with model details augmented by the
            selected model's parameters and cross-validation score.
        """
        optimizer_best_model_params_dict = dict()
        with open(f"tmp/optimizer_best_model_params_dict.pickle", "wb") as f:
            pickle.dump(optimizer_best_model_params_dict, f)

        study = self.optimize(direction=direction, n_trials=n_trials)
        best_study_params = study.best_params
        best_trial = study.best_trial
        imputer = best_study_params["imputer"]
        outlier_method = best_study_params["outlier_method"]
        feature_engineering_scaler = best_study_params[
            "feature_engineering_scaler"
        ]
        oh_encoder_min_frequency = best_study_params["oh_encoder_min_frequency"]
        model_to_train = best_study_params["model_to_train"]

        with open(f"tmp/optimizer_best_model_params_dict.pickle", "rb") as f:
            optimizer_best_model_params_dict = pickle.load(f)
        best_iteration_result = optimizer_best_model_params_dict[
            best_trial.number
        ]

        final_run_result = self.run_pipeline(
            best_trial,
            imputer,
            outlier_method,
            feature_engineering_scaler,
            oh_encoder_min_frequency,
            model_to_train,
        )
        best_study_params["model_to_train"] = (
            best_study_params["model_to_train"],
            (
                best_iteration_result[constants.BEST_PARAMETERS_NAME],
                best_iteration_result[constants.TRAIN_CV_SCORE_NAME],
            ),
        )
        return best_study_params
