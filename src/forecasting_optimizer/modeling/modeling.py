# Copyright 2023 Anirudh Singh
# SPDX-License-Identifier: BSD-3-Clause

"""Orchestration for fitting and evaluating a configured forecasting model."""

import importlib
import logging
from datetime import datetime

import numpy as np

from forecasting_optimizer import constants
from forecasting_optimizer import experiment_settings
from forecasting_optimizer import framework_settings
from forecasting_optimizer.modeling import modeling_constants
from forecasting_optimizer.modeling.model_classes import MODEL_CLASSES
from forecasting_optimizer.util import utility_functions as uf

experiment_constants = importlib.import_module(
    "forecasting_optimizer.experiment_configs."
    f"{experiment_settings.EXPERIMENT_CONSTANTS_MODULE_NAME}"
)


class Modeling:
    """Coordinates training, evaluation, and persistence for one model.

    Attributes:
        experiment_id: Identifier used for experiment artifacts.
        trial: Optimization trial associated with the model run.
        data: Tuple containing train, validation, and test datasets.
        feature_engineering_scaler: Optional scaler selected upstream.
        model: Registered name of the model selected for training.
    """

    def __init__(
        self,
        experiment_id,
        trial,
        data=None,
        feature_engineering_scaler=None,
        model=None,
    ):
        """Initializes the modeling stage.

        Args:
            experiment_id: Identifier used to load and persist experiment data.
            trial: Optimization trial associated with this model run.
            data: Optional ``(train, validation, test)`` dataset tuple. The
                retained no-data fallback passes an incompatible keyword to
                ``_load_data``; supplying the tuple avoids that legacy path.
            feature_engineering_scaler: Optional scaler already selected by the
                feature-engineering stage.
            model: Optional model name. The framework default is used when this
                is omitted.
        """
        self.experiment_id = experiment_id
        self.trial = trial

        if data is None:
            self.data = self._load_data(experiment_id=self.experiment_id)
        else:
            self.data = data

        self.feature_engineering_scaler = feature_engineering_scaler

        if model is None:
            self.model = framework_settings.DEFAULT_MODEL_TO_TRAIN
        else:
            self.model = model

        self._models_to_train = [self.model]

    def _load_data(self):
        """Loads feature-engineered train, validation, and test datasets.

        Returns:
            A tuple containing the persisted train, validation, and test
            DataFrames.
        """
        train = uf.read_df(
            self.experiment_id,
            name=(
                f"{constants.FEATURE_ENGINEERING_NAME}_{constants.TRAIN_NAME}"
            ),
        )
        val = uf.read_df(
            self.experiment_id,
            name=f"{constants.FEATURE_ENGINEERING_NAME}_{constants.VAL_NAME}",
        )
        test = uf.read_df(
            self.experiment_id,
            name=f"{constants.FEATURE_ENGINEERING_NAME}_{constants.TEST_NAME}",
        )
        return train, val, test

    def _train_one_model(self, model_name):
        """Fits and evaluates one configured model.

        Args:
            model_name: Key used to select a model implementation.

        Returns:
            A tuple containing train, validation, and test metrics, followed by
            the best hyperparameters and their cross-validation score.
        """
        _model_class = MODEL_CLASSES[model_name]
        model = _model_class(
            data=self.data,
            target_col_name=experiment_constants.TARGET_COL,
            feature_engineering_scaler=self.feature_engineering_scaler,
            experiment_id=self.experiment_id,
        )
        model, best_params, best_params_cv_score = model.fit()
        train_metric, val_metric, test_metric = model.evaluate()
        logging.info(
            f"Training performance: {train_metric} \n Val performance: "
            f"{val_metric} \n Test performance: {test_metric}"
        )
        return (
            train_metric,
            val_metric,
            test_metric,
            best_params,
            best_params_cv_score,
        )

    def execute_modeling(self):
        """Runs the configured model and persists its performance summary.

        Returns:
            A mapping of best parameters and train, validation, test, and
            cross-validation metrics.
        """
        (
            train_metric,
            val_metric,
            test_metric,
            best_params,
            best_params_cv_score,
        ) = self._train_one_model(
            model_name=self.model,
        )
        model_performance = {
            constants.BEST_PARAMETERS_NAME: best_params,
            constants.TRAIN_NAME: train_metric,
            constants.VAL_NAME: val_metric,
            constants.TEST_NAME: test_metric,
            constants.TRAIN_CV_SCORE_NAME: best_params_cv_score,
        }
        model_key = ""
        for k, v in model_performance[constants.BEST_PARAMETERS_NAME].items():
            if isinstance(v, float):
                v = np.round(v, 4)
            # Distinguish parameter combinations; identical combinations can
            # still resolve to the same artifact name.
            model_key = f"{model_key}_{k}_{v}"

        uf.save_data(
            model_performance,
            self.experiment_id,
            name=(
                f"{constants.MODELING_NAME}_{model_key}_"
                f"{constants.MODEL_PERFORMANCE_RESULT_NAME}"
            ),
        )
        return model_performance
