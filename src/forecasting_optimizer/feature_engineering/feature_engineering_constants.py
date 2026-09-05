# Copyright 2023 Anirudh Singh
# SPDX-License-Identifier: BSD-3-Clause

"""Experiment-scoped dependencies for feature engineering configuration."""

import importlib

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from forecasting_optimizer import experiment_settings

experiment_constants = importlib.import_module(
    "forecasting_optimizer.experiment_configs."
    f"{experiment_settings.EXPERIMENT_CONSTANTS_MODULE_NAME}"
)
