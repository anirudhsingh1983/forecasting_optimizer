# Copyright 2023 Anirudh Singh
# SPDX-License-Identifier: BSD-3-Clause

"""Run the configured forecasting optimization experiment."""

import importlib
import logging
from datetime import datetime

import pandas as pd

from forecasting_optimizer import experiment_settings
from forecasting_optimizer.optimization.optimizer import Optimizer
from forecasting_optimizer.util import utility_functions as uf

# These process-wide options keep diagnostic frames readable during long runs.
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


def main():
    """Run optimization and persist the best-parameter summary."""
    start = datetime.now()
    log = logging.getLogger(__name__)

    log.info(f"Backtesting started at: {str(start)}")

    experiment_id = "e00eb1e7-cc1b-417a-87fe-a8dfff2a2c29"
    # Use the configuration-generated identifier when that lifecycle is desired.
    # experiment_id = experiment_constants.EXPERIMENT_ID

    op = Optimizer(experiment_id=experiment_id)
    best_pipeline = op.execute_optimization(direction="minimize", n_trials=200)
    print(best_pipeline)
    uf.save_data(best_pipeline, experiment_id, name=f"final_result")


if __name__ == "__main__":
    main()
