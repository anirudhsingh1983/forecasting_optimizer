import importlib
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest

import constants
import experiment_settings
import framework_settings
from eda import eda_constants
from util import utility_functions as uf

experiment_constants = importlib.import_module(
    f"experiment_configs.{experiment_settings.EXPERIMENT_CONSTANTS_MODULE_NAME}")

start = datetime.now()
log = logging.getLogger(__name__)

log.info(f"Backtesting started at: {str(start)}")

TARGET_COL = experiment_constants.TARGET_COL

try:
    REMOVE_OUTLIERS = experiment_constants.REMOVE_OUTLIERS
    if experiment_constants.REMOVE_OUTLIERS is None:
        REMOVE_OUTLIERS = eda_constants.DEFAULT_REMOVE_OUTLIERS
except:
    REMOVE_OUTLIERS = eda_constants.DEFAULT_REMOVE_OUTLIERS

try:
    OUTLIER_METHOD = experiment_constants.OUTLIER_METHOD
    if (OUTLIER_METHOD is None) | (OUTLIER_METHOD not in constants.OUTLIER_METHODS):
        OUTLIER_METHOD = framework_settings.DEFAULT_OUTLIER_METHOD
except:
    OUTLIER_METHOD = framework_settings.DEFAULT_OUTLIER_METHOD

try:
    MISSING_VALUE_THRESHOLD = experiment_constants.MISSING_VALUE_THRESHOLD
    if experiment_constants.MISSING_VALUE_THRESHOLD is None:
        MISSING_VALUE_THRESHOLD = eda_constants.DEFAULT_MISSING_VALUE_THRESHOLD
except:
    MISSING_VALUE_THRESHOLD = eda_constants.DEFAULT_MISSING_VALUE_THRESHOLD


def _load_data(experiment_id):
    data = uf.read_df(experiment_id, name=constants.LANDING_DATA_NAME)
    return data


def _fill_timeseries_gaps(data):
    data.index = pd.to_datetime(data.index)
    ts_interval = pd.to_timedelta(arg=experiment_constants.TIMESERIES_INTERVAL_VALUE,
                                  unit=experiment_constants.TIMESERIES_INTERVAL_UNIT)
    full_index = pd.date_range(start=pd.to_datetime(data.index.min()), end=pd.to_datetime(data.index.max()),
                               freq=ts_interval)
    full_index = full_index.union(data.index)
    data = data.reindex(full_index)
    data = data.sort_index(ascending=True)
    data[experiment_constants.TARGET_COL] = data[experiment_constants.TARGET_COL].fillna(
        experiment_constants.TIMESERIES_TARGET_FILL_GAP_VALUE)
    return data


def _generate_plots(data):
    target = data[TARGET_COL]
    exo_vars = data.drop(columns=TARGET_COL).columns
    exo_data = data[exo_vars]

    # plot target
    plt.plot(target, label=constants.TARGET_NAME)
    plt.legend()
    plt.show()

    # plot exogenous variables
    if exo_data.shape[1] > 0:
        exo_continuous = exo_data.select_dtypes(include=np.number)
        exo_categorical = exo_data.select_dtypes(exclude=np.number)

        for col in exo_continuous.columns:
            plt.plot(exo_continuous[col], label=f"{col}")
            plt.show()

        for col in exo_categorical.columns:
            plt.hist(exo_categorical[col], label=f"{col}")
            plt.show()


def _calculate_missing_values(data):
    mv = data.isna().sum() / len(data)
    mv_high = mv[mv > MISSING_VALUE_THRESHOLD]
    if len(mv_high) > 0:
        logging.warning(
            f"The columns having higher than {MISSING_VALUE_THRESHOLD * 100}% missing values are: {mv_high}")
    return mv, mv_high


def _check_target_stationarity(data):
    return None


def _identify_outliers(data):
    if OUTLIER_METHOD == constants.OUTLIER_METHOD_BOXPLOT_NAME:
        pass
    elif OUTLIER_METHOD == constants.OUTLIER_METHOD_ZSCORE_NAME:
        pass
    elif OUTLIER_METHOD == constants.OUTLIER_METHOD_SVMONECLASS_NAME:
        pass
    elif OUTLIER_METHOD == constants.OUTLIER_METHOD_ISOLATIONFOREST_NAME:
        clf = IsolationForest(n_estimators=100, warm_start=True)
        clf.fit(data)

        y_pred = clf.predict(data)
        y_pred = pd.Series(y_pred)
        ol_count = y_pred.value_counts()[-1]  # -1 indicates outlier detected by isolation forest
        logging.warning(f"The total number of outliers are: {ol_count}")
        mask = (y_pred == -1).values
        data_ol = data.loc[mask, :]
        data_norm = data.loc[~mask, :]
        return data_norm, data_ol


def execute_eda(experiment_id):
    data = _load_data(experiment_id=experiment_id)
    data = _fill_timeseries_gaps(data)
    _generate_plots(data=data)
    mv, mv_high = _calculate_missing_values(data=data)
    is_stationary = _check_target_stationarity(data)
    data_norm, data_ol = _identify_outliers(data)
    if REMOVE_OUTLIERS:
        data = data_norm

    uf.save_df(data, experiment_id, name=f"{constants.EDA_NAME}")

    return mv, mv_high, is_stationary, data_norm, data_ol
