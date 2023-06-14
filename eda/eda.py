import importlib
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from datetime import date, time, datetime

import constants
import experiment_settings
import framework_settings
from util import utility_functions as uf

experiment_constants = importlib.import_module(
    f"experiment_configs.{experiment_settings.EXPERIMENT_CONSTANTS_MODULE_NAME}")

start = datetime.now()
log = logging.getLogger(__name__)

TARGET_COL = experiment_constants.TARGET_COL

try:
    REMOVE_OUTLIERS = experiment_constants.REMOVE_OUTLIERS
    if experiment_constants.REMOVE_OUTLIERS is None:
        REMOVE_OUTLIERS = framework_settings.DEFAULT_REMOVE_OUTLIERS
except:
    REMOVE_OUTLIERS = framework_settings.DEFAULT_REMOVE_OUTLIERS

try:
    OUTLIER_METHOD = experiment_constants.OUTLIER_METHOD
    if (OUTLIER_METHOD is None) | (OUTLIER_METHOD not in constants.OUTLIER_METHODS):
        OUTLIER_METHOD = framework_settings.DEFAULT_OUTLIER_METHOD
except:
    OUTLIER_METHOD = framework_settings.DEFAULT_OUTLIER_METHOD

try:
    MISSING_VALUE_THRESHOLD = experiment_constants.MISSING_VALUE_THRESHOLD
    if experiment_constants.MISSING_VALUE_THRESHOLD is None:
        MISSING_VALUE_THRESHOLD = framework_settings.DEFAULT_MISSING_VALUE_THRESHOLD
except:
    MISSING_VALUE_THRESHOLD = framework_settings.DEFAULT_MISSING_VALUE_THRESHOLD


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
    gap_idx = set(full_index) - set(data.index)
    gap_idx = pd.Index(gap_idx).sort_values()
    data = data.reindex(full_index)
    data = data.sort_index(ascending=True)

    if experiment_constants.TIMESERIES_TARGET_FILL_GAP_VALUE in ['backfill', 'bfill', 'ffill']:
        data[experiment_constants.TARGET_COL].loc[gap_idx] = data[experiment_constants.TARGET_COL].fillna(
            method = experiment_constants.TIMESERIES_TARGET_FILL_GAP_VALUE).loc[gap_idx]
    else:
        data[experiment_constants.TARGET_COL].loc[gap_idx] = data[experiment_constants.TARGET_COL].fillna(
            value=experiment_constants.TIMESERIES_TARGET_FILL_GAP_VALUE).loc[gap_idx]
    return data


def _generate_plots(data):
    target = data[TARGET_COL]
    exo_vars = data.drop(columns=TARGET_COL).columns
    exo_data = data[exo_vars]

    try:
        imputing_val_dict = experiment_constants.IMPUTING_VAL_DICT
    except:
        logging.warning(f"No imputing value dictionary in the experiment setting. So, the framework's default imputing values will be used.")
        imputing_val_dict = dict()

    # plot target
    try:
        imputing_value = imputing_val_dict[TARGET_COL]
    except:
        imputing_value = framework_settings.DEFAULT_CONTINUOUS_MISSING_IMPUTATION_VALUE

    if imputing_value in ['backfill', 'bfill', 'ffill']:
        target = target.fillna(method = imputing_value)
    else:
        target = target.fillna(value=imputing_value)


    # target.loc[target.isna()] = imputing_value
    plt.plot(target, label=constants.TARGET_NAME)
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

    # plot exogenous variables
    if exo_data.shape[1] > 0:
        exo_continuous = exo_data.select_dtypes(include=np.number)
        exo_categorical = exo_data.select_dtypes(exclude=np.number)

        for col in exo_continuous.columns:
            try:
                imputing_value = imputing_val_dict[col]
            except:
                imputing_value = framework_settings.DEFAULT_CONTINUOUS_MISSING_IMPUTATION_VALUE

            if imputing_value in ['backfill', 'bfill', 'ffill']:
                exo_continuous.loc[:, col] = exo_continuous[col].fillna(method=imputing_value)
            else:
                exo_continuous.loc[:, col] = exo_continuous[col].fillna(value=imputing_value)
            # exo_continuous.loc[exo_continuous[col].isna(), col] = imputing_value

            plt.plot(exo_continuous[col], label=f"{col}")
            plt.xticks(rotation=90)
            plt.legend()
            plt.show()

        for col in exo_categorical.columns:
            try:
                imputing_value = imputing_val_dict[col]
            except:
                imputing_value = framework_settings.DEFAULT_CATEGORICAL_MISSING_IMPUTATION_VALUE

            if imputing_value in ['backfill', 'bfill', 'ffill']:
                exo_categorical.loc[:, col] = exo_categorical[col].fillna(method=imputing_value)
            else:
                exo_categorical.loc[:, col] = exo_categorical[col].fillna(value=imputing_value)
            # exo_categorical.loc[exo_categorical[col].isna(), col] = imputing_value

            types = [date, time, datetime]
            if type(exo_categorical[col].iloc[0]) in types:
                continue # skip plotting dates. Newer versions of matplotlib do not allow non-timestamp values in timestamps even if the timestamps are converted to string. To avoid this issue, dates are being avoided while plotting.
            else:
                plt.hist(exo_categorical[col], label=f"{col}")
                plt.legend()
                plt.xticks(rotation=90)
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


def execute_eda(experiment_id, data, plots=True):
    data = _fill_timeseries_gaps(data)
    mv, mv_high = _calculate_missing_values(data=data)
    if plots:
        _generate_plots(data=data)
    is_stationary = _check_target_stationarity(data)
    # data_norm, data_ol = _identify_outliers(data)
    uf.save_df(data, experiment_id, name=f"{constants.EDA_NAME}")
    # return data, mv, mv_high, is_stationary, data_norm, data_ol
    return data, mv, mv_high, is_stationary
