# Copyright 2023 Anirudh Singh
# SPDX-License-Identifier: BSD-3-Clause

"""Load and persist the raw data for a forecasting experiment."""

from forecasting_optimizer import constants
from forecasting_optimizer.util import utility_functions as uf


def execute_data_landing(experiment_id, data_loading_function):
    """Run a data loader and save its result as the experiment's landed data.

    Args:
        experiment_id: Identifier used to namespace the saved artifact.
        data_loading_function: Zero-argument callable that returns the raw data.

    Returns:
        The data returned by ``data_loading_function``.
    """
    data = data_loading_function()
    uf.save_df(data, experiment_id, name=constants.LANDING_DATA_NAME)
    return data
