import importlib
import logging
from datetime import datetime

import pandas as pd

import constants
import framework_settings
import experiment_settings
from data_preprocessing import data_preprocessing_constants
from util import utility_functions as uf

experiment_constants = importlib.import_module(
    f"experiment_configs.{experiment_settings.EXPERIMENT_CONSTANTS_MODULE_NAME}")


class DataPreprocessing():
    def __init__(self, experiment_id, trial, data=None, imputer=None):
        self.experiment_id = experiment_id
        self.trial = trial
        if data is None:
            self.data = self._load_data(experiment_id=experiment_id)
        else:
            self.data = data

        # Note: preprocesing_function will have a mandatory argument 'trial', so it can have optimizable arguments to leverage various data preprocessing ways to get the best outcome
        try:
            preprocesing_function = experiment_constants.PREPROCESSING_FUNCTION
            if preprocesing_function is None:
                preprocesing_function = lambda trial, x: x
        except:
            preprocesing_function = experiment_constants.PREPROCESSING_FUNCTION
        self.preprocesing_function = preprocesing_function

        try:
            train_val_split_date = experiment_constants.TRAIN_VAL_SPLIT_DATE
            if train_val_split_date is None:
                train_val_split_date = self._get_default_train_val_split_date()
        except:
            train_val_split_date = self._get_default_train_val_split_date()

        try:
            val_test_split_date = experiment_constants.VAL_TEST_SPLIT_DATE
            if val_test_split_date is None:
                val_test_split_date = self._get_default_val_test_split_date()
        except:
            val_test_split_date = self._get_default_val_test_split_date()

        if imputer is None:
            self.imputer = framework_settings.DEFAULT_IMPUTER
        else:
            self.imputer = imputer
        self.preprocesing_function = preprocesing_function
        self.train_val_split_date = train_val_split_date
        self.val_test_split_date = val_test_split_date

    def _get_default_train_val_split_date(self):
        horizon = self.data.index.max() - self.data.index.min()
        default_train_val_split_date = self.data.index.max() - 3 * horizon / 10 # leaves 20% (30% - 10% of test set) for val set
        return default_train_val_split_date

    def _get_default_val_test_split_date(self):
        horizon = self.data.index.max() - self.data.index.min()
        default_val_test_split_date = self.data.index.max() - 1 * horizon / 10 # leaves ~10% data for test set
        return default_val_test_split_date

    def _load_data(self, experiment_id):
        data = uf.read_df(experiment_id, name=constants.EDA_NAME)
        return data

    def _save_processed_data(self):
        uf.save_df(self.train, self.experiment_id, name=f"{constants.DATA_PREPROCESSING_NAME}_{constants.TRAIN_NAME}")
        uf.save_df(self.val, self.experiment_id, name=f"{constants.DATA_PREPROCESSING_NAME}_{constants.VAL_NAME}")
        uf.save_df(self.test, self.experiment_id, name=f"{constants.DATA_PREPROCESSING_NAME}_{constants.TEST_NAME}")

    def _split_data(self):
        data = self.data
        data.index = pd.to_datetime(data.index)
        self.train = data[data.index < self.train_val_split_date]
        self.val = data[(data.index >= self.train_val_split_date) & (data.index < self.val_test_split_date)]
        self.test = data[data.index >= self.val_test_split_date]

    def _treat_missing_values(self):
        features = self.train.drop(columns=[experiment_constants.TARGET_COL]).columns
        if self.imputer in ['backfill', 'bfill', 'ffill']:
            self.train.loc[:, features] = self.train[features].fillna(method=self.imputer)
            if len(self.val) > 0:
                self.val.loc[:, features] = self.val[features].fillna(method=self.imputer)
            if len(self.test) > 0:
                self.test.loc[:, features] = self.test[features].fillna(method=self.imputer)
        else:
            self.imputer.fit(self.train)
            self.train = pd.DataFrame(data=self.imputer.transform(self.train), index=self.train.index, columns=self.train.columns)
            if len(self.val) > 0:
                self.val = pd.DataFrame(data=self.imputer.transform(self.val), index=self.val.index, columns=self.val.columns)
            if len(self.test) > 0:
                self.test = pd.DataFrame(data=self.imputer.transform(self.test), index=self.test.index, columns=self.test.columns)


    def execute_preprocessing(self):
        # The _preprocesing_function should do the following:
        # General data processing (may include scaling/transformations/etc.)
        # Data filtering, as needed
        # Feature engineering pre train-val-test split
        self.data = self.preprocesing_function(self.trial, self.data)
        self._split_data()
        self._treat_missing_values()
        self._save_processed_data()
        return self.train, self.val, self.test
