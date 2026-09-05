# Copyright 2023 Anirudh Singh
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime settings for the end-to-end logistics forecast."""

from forecasting_optimizer.projects.e2e import constants as const

# Application settings.
READ_DATA_LOCAL = True
OH_ENCODER_MIN_FREQUENCY = 10
DEFAULT_SPLIT_DAYS = 90
MAX_TRAINING_DAYS = 360
MIN_PREV_NODE_TRAINING_RECORDS = 100  # Previous-node volume models only.
MIN_TRAINING_RECORDS = 50  # Transit and dwell models only.
TRAIN_WITH_WEIGHTS = False
EVAL_TEST_WITH_WEIGHTS = True
XGBOOST_MAX_DEPTH = 2
TRANSIT_STD_DEVS = 2
DWELL_STD_DEVS = 2
MISSING_VAL_WARNING_PERCENT = 5
MAX_TRANSIT_DAYS = 300
MAX_DWELL_DAYS = 100
PREV_VOL_LEG_MODEL_TYPE = "xgboost"  # options: ['xgboost']
TRANSIT_LEG_MODEL_TYPE = "xgboost"  # options: ['xgboost', 'linear']
DWELL_LEG_MODEL_TYPE = "xgboost"  # options: ['xgboost', 'linear']
PREV_VOL_UNIT_MODEL_TYPE = "xgboost"  # options: ['xgboost']
TRANSIT_UNIT_MODEL_TYPE = "xgboost"  # options: ['xgboost', 'linear']
DWELL_UNIT_MODEL_TYPE = "xgboost"  # options: ['xgboost', 'linear']
GLOBAL_OUTLIERS_REMOVAL = {
    # Legacy options intended for external consumers of these settings.
    const.PREV_NODE_VOL_MODEL: None,
    const.TRANSIT_TIME_MODEL: None,
    const.DWELL_TIME_MODEL: None,
}
MODEL_OUTLIERS_REMOVAL = {
    # Retain historical values even though this repository does not read them.
    const.PREV_NODE_VOL_MODEL: None,
    const.TRANSIT_TIME_MODEL: "boxplot",
    const.DWELL_TIME_MODEL: "boxplot",
}
# Entries may be selected from day, dayofweek, week, and month.
BASELINE_GROUPING_FEATURES = ["week", "dayofweek"]
CONSOLIDATE_ORIGIN_PORTS = False

# Flowback settings.
NODES_TO_SKIP_FLOWBACK = {
    const.ACF_LOCATION_NAME_COL: [const.NO_CFS_STR],
    const.ORIGIN_PORT_NAME_COL: [],
    const.DESTINATION_PORT_NAME_COL: [],
    const.FC_KEY: [],
}
