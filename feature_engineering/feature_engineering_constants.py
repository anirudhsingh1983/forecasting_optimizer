import importlib

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

import experiment_settings

experiment_constants = importlib.import_module(
    f"experiment_configs.{experiment_settings.EXPERIMENT_CONSTANTS_MODULE_NAME}")

