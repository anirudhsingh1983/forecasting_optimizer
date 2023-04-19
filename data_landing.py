import importlib

import constants
import experiment_settings
from util import utility_functions as uf

experiment_constants = importlib.import_module(
    f"experiment_configs.{experiment_settings.EXPERIMENT_CONSTANTS_MODULE_NAME}")

_data_loading_function = experiment_constants.DATA_LOADING_FUNCTION


def execute_data_landing(experiment_id):
    data = _data_loading_function()
    uf.save_df(data, experiment_id, name=constants.LANDING_DATA_NAME)
