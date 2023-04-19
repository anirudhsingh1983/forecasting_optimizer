import importlib
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import constants
import experiment_settings
import framework_settings
from feature_engineering import feature_engineering_constants
from util import utility_functions as uf

experiment_constants = importlib.import_module(
    f"experiment_configs.{experiment_settings.EXPERIMENT_CONSTANTS_MODULE_NAME}")

start = datetime.now()
log = logging.getLogger(__name__)

_feature_development_function = feature_engineering_constants.FEATURE_ENGINEERING_FUNCTION

try:
    OUTLIER_METHOD = experiment_constants.OUTLIER_METHOD
    if (OUTLIER_METHOD is None) | (OUTLIER_METHOD not in constants.OUTLIER_METHODS):
        OUTLIER_METHOD = framework_settings.DEFAULT_OUTLIER_METHOD
except:
    OUTLIER_METHOD = framework_settings.DEFAULT_OUTLIER_METHOD


def _load_data(experiment_id):
    train = uf.read_df(experiment_id, name=f"{constants.DATA_PREPROCESSING_NAME}_{constants.TRAIN_NAME}")
    val = uf.read_df(experiment_id, name=f"{constants.DATA_PREPROCESSING_NAME}_{constants.VAL_NAME}")
    test = uf.read_df(experiment_id, name=f"{constants.DATA_PREPROCESSING_NAME}_{constants.TEST_NAME}")
    return train, val, test


def _save_processed_data(data, experiment_id):
    train, val, test = data
    uf.save_df(train, experiment_id, name=f"{constants.FEATURE_ENGINEERING_NAME}_{constants.TRAIN_NAME}")
    uf.save_df(val, experiment_id, name=f"{constants.FEATURE_ENGINEERING_NAME}_{constants.VAL_NAME}")
    uf.save_df(test, experiment_id, name=f"{constants.FEATURE_ENGINEERING_NAME}_{constants.TEST_NAME}")


def _generate_features(train, val, test):
    train = _feature_development_function(train)
    val = _feature_development_function(val)
    test = _feature_development_function(test)
    return train, val, test


def _treat_categorical_features(train, val, test):
    encoder = feature_engineering_constants.CATEGORICAL_FEATURE_ENCODER
    categorical_features = train.select_dtypes(
        include=constants.OBJECT_NAME).columns.tolist() + feature_engineering_constants.CATEGORICAL_ENGINEERED_FEATURES
    categorical_features = list(set(categorical_features).intersection(train.columns))  # to remove duplicate features
    encoder.fit(train[categorical_features])
    encoded_columns = [x if (y is None) else ([e for e in x if e not in y] + ['infrequent_val']) for x, y in
                       tuple(zip(encoder.categories_, encoder.infrequent_categories_))]
    encoded_columns = dict(zip(categorical_features, encoded_columns))
    encoded_columns = [[f"{col}_{val}" for val in list(encoded_values)] for col, encoded_values in
                       encoded_columns.items()]
    encoded_columns = np.concatenate(encoded_columns)
    train_encoded = pd.DataFrame(
        data=encoder.transform(train[categorical_features]).toarray(),
        index=train.index,
        columns=encoded_columns,
    )
    val_encoded = pd.DataFrame(
        data=encoder.transform(val[categorical_features]).toarray(),
        index=val.index,
        columns=encoded_columns,
    )
    test_encoded = pd.DataFrame(
        data=encoder.transform(test[categorical_features]).toarray(),
        index=test.index,
        columns=encoded_columns,
    )
    train = pd.concat([train.drop(columns=categorical_features), train_encoded], axis=1)
    val = pd.concat([val.drop(columns=categorical_features), val_encoded], axis=1)
    test = pd.concat([test.drop(columns=categorical_features), test_encoded], axis=1)
    return train, val, test


def _scale_data(train, val, test):
    scaler = feature_engineering_constants.FEATURE_ENGINEERING_SCALER
    scaler.fit(train)
    train = pd.DataFrame(
        data=scaler.transform(train),
        index=train.index,
        columns=train.columns,
    )
    val = pd.DataFrame(
        data=scaler.transform(val),
        index=val.index,
        columns=val.columns,
    )
    test = pd.DataFrame(
        data=scaler.transform(test),
        index=test.index,
        columns=test.columns,
    )
    return train, val, test


def _identify_isolation_forest_outliers(data, clf, data_name='data'):
    y_pred = clf.predict(data)
    y_pred = pd.Series(y_pred)
    if (-1 in y_pred.values):
        ol_count = y_pred.value_counts()[-1]  # -1 indicates outlier detected by isolation forest
    else:
        ol_count = 0
    logging.warning(f"The total number of outliers in {data_name} are: {ol_count}")
    mask = (y_pred == -1).values
    data_ol = data.loc[mask, :]
    data_norm = data.loc[~mask, :]
    return data_norm, data_ol


def _remove_outliers(train, val, test):
    if OUTLIER_METHOD == constants.OUTLIER_METHOD_BOXPLOT_NAME:
        pass
    elif OUTLIER_METHOD == constants.OUTLIER_METHOD_ZSCORE_NAME:
        pass
    elif OUTLIER_METHOD == constants.OUTLIER_METHOD_SVMONECLASS_NAME:
        pass
    elif OUTLIER_METHOD == constants.OUTLIER_METHOD_ISOLATIONFOREST_NAME:
        clf = IsolationForest(n_estimators=100, warm_start=True)
        clf.fit(train)
        train, _ = _identify_isolation_forest_outliers(train, clf, data_name='train')
        val, _ = _identify_isolation_forest_outliers(val, clf, data_name='val')
        test, _ = _identify_isolation_forest_outliers(test, clf, data_name='test')

    return train, val, test


def execute_feature_engineering(experiment_id):
    train, val, test = _load_data(experiment_id=experiment_id)
    train, val, test = _generate_features(train, val, test)
    train, val, test = _treat_categorical_features(train, val, test)
    train, val, test = _scale_data(train, val, test)
    train, val, test = _remove_outliers(train, val, test)
    _save_processed_data((train, val, test), experiment_id)
