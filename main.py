import importlib
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

import data_landing
import experiment_settings
from data_preprocessing import data_preprocessing
from eda import eda
from feature_engineering import feature_engineering
from modeling import modeling
from optimization.optimizer import Optimizer

pd.set_option('display.expand_frame_repr', True)
pd.set_option('display.max_columns', None)  # Set to None for unlimited number of output rows.
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_info_columns', 500)
pd.set_option('display.max_rows', None)  # Set to None for unlimited number of output rows.
pd.set_option('display.width', 500)  # Width of the display in characters.

experiment_constants = importlib.import_module(
    f"experiment_configs.{experiment_settings.EXPERIMENT_CONSTANTS_MODULE_NAME}")


# begin experiment
start = datetime.now()
log = logging.getLogger(__name__)

log.info(f"Backtesting started at: {str(start)}")

experiment_id = 'e00eb1e7-cc1b-417a-87fe-a8dfff2a2c29'

op = Optimizer(experiment_id=experiment_id)
# study = op.optimize(direction='minimize', n_trials=2)
best_pipeline = op.execute_optimization(direction='minimize', n_trials=2)
best_pipeline

# import constants
# from util import utility_functions as uf
# model_performances = uf.read_data(experiment_id, name=f"{constants.MODELING_NAME}_{constants.MODEL_PERFORMANCE_RESULT_NAME}")
