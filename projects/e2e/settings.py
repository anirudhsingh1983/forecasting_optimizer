
from projects.e2e import constants as const

# app settings
READ_DATA_LOCAL = True
OH_ENCODER_MIN_FREQUENCY = 10
DEFAULT_SPLIT_DAYS = 90
MAX_TRAINING_DAYS = 360
MIN_PREV_NODE_TRAINING_RECORDS = 100  # for prev node volume models only
MIN_TRAINING_RECORDS = 50 # for transit and dwell models only
TRAIN_WITH_WEIGHTS = False
EVAL_TEST_WITH_WEIGHTS = True
XGBOOST_MAX_DEPTH = 2
TRANSIT_STD_DEVS = 2
DWELL_STD_DEVS = 2
MISSING_VAL_WARNING_PERCENT = 5
MAX_TRANSIT_DAYS = 300
MAX_DWELL_DAYS = 100
PREV_VOL_LEG_MODEL_TYPE = 'xgboost'  # options: ['xgboost']
TRANSIT_LEG_MODEL_TYPE = 'xgboost'  # options: ['xgboost', 'linear']
DWELL_LEG_MODEL_TYPE = 'xgboost'  # options: ['xgboost', 'linear']
PREV_VOL_UNIT_MODEL_TYPE = 'xgboost' # options: ['xgboost']
TRANSIT_UNIT_MODEL_TYPE = 'xgboost' # options: ['xgboost', 'linear']
DWELL_UNIT_MODEL_TYPE = 'xgboost' # options: ['xgboost', 'linear']
GLOBAL_OUTLIERS_REMOVAL = {
    const.PREV_NODE_VOL_MODEL: None,  # possible dict values: [None, 'isolation_forest']
    const.TRANSIT_TIME_MODEL: None,  # possible dict values: [None, 'boxplot', 'isolation_forest']
    const.DWELL_TIME_MODEL: None,  # possible dict values: [None, 'boxplot', 'isolation_forest']
}
MODEL_OUTLIERS_REMOVAL = {
    const.PREV_NODE_VOL_MODEL: None,  # possible dict values: [None, 'isolation_forest']
    const.TRANSIT_TIME_MODEL: 'boxplot',  # possible dict values: [None, 'boxplot', 'isolation_forest']
    const.DWELL_TIME_MODEL: 'boxplot',  # possible dict values: [None, 'boxplot', 'isolation_forest']
}
BASELINE_GROUPING_FEATURES = ['week', 'dayofweek']  # can be among ['day', 'dayofweek', 'week', 'month']
CONSOLIDATE_ORIGIN_PORTS = False

# Flowback
NODES_TO_SKIP_FLOWBACK = {
    const.ACF_LOCATION_NAME_COL: [const.NO_CFS_STR],
    const.ORIGIN_PORT_NAME_COL: [],
    const.DESTINATION_PORT_NAME_COL: [],
    const.FC_KEY: [],
}
