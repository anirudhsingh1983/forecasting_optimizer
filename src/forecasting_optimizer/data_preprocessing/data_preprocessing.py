# Copyright 2023 Anirudh Singh
# SPDX-License-Identifier: BSD-3-Clause

"""Split experiment data and apply configured missing-value treatment."""

import importlib
import logging
from datetime import datetime

import pandas as pd

from forecasting_optimizer import constants
from forecasting_optimizer import experiment_settings
from forecasting_optimizer import framework_settings
from forecasting_optimizer.data_preprocessing import (
    data_preprocessing_constants,
)
from forecasting_optimizer.util import utility_functions as uf

experiment_constants = importlib.import_module(
    "forecasting_optimizer.experiment_configs."
    f"{experiment_settings.EXPERIMENT_CONSTANTS_MODULE_NAME}"
)


class DataPreprocessing:
    """Prepare train, validation, and test data for feature engineering.

    Attributes:
        experiment_id: str or UUID identifier for experiment artifacts.
        trial: object representing the optimization trial passed to callbacks.
        data: pandas DataFrame containing source data to preprocess.
        imputer: str strategy or transformer used for missing values.
        preprocesing_function: Callable applied before splitting.
        train_val_split_date: datetime-like exclusive training upper bound.
        val_test_split_date: datetime-like inclusive test lower bound.
        train: pandas DataFrame populated during preprocessing.
        val: pandas DataFrame populated during preprocessing.
        test: pandas DataFrame populated during preprocessing.
    """

    def __init__(self, experiment_id, trial, data=None, imputer=None):
        """Initialize preprocessing for an experiment.

        Args:
            experiment_id: str or UUID identifier used for artifacts.
            trial: object representing the optimization trial passed to the
                experiment callback.
            data: Optional pandas DataFrame. If omitted, the EDA artifact is
                loaded.
            imputer: Optional str missing-value strategy or transformer. If
                omitted, the framework default is used.
        """
        self.experiment_id = experiment_id
        self.trial = trial
        if data is None:
            self.data = self._load_data(experiment_id=experiment_id)
        else:
            self.data = data

        # Passing the trial lets experiments optimize preprocessing choices.
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
        """Calculate the default boundary after the first 70% of the horizon.

        Returns:
            The default training-to-validation split timestamp.
        """
        horizon = self.data.index.max() - self.data.index.min()
        # Reserving the final 30% leaves 20% for validation and 10% for test.
        default_train_val_split_date = self.data.index.max() - 3 * horizon / 10
        return default_train_val_split_date

    def _get_default_val_test_split_date(self):
        """Calculate the default boundary before the final 10% of the horizon.

        Returns:
            The default validation-to-test split timestamp.
        """
        horizon = self.data.index.max() - self.data.index.min()
        default_val_test_split_date = self.data.index.max() - 1 * horizon / 10
        return default_val_test_split_date

    def _load_data(self, experiment_id):
        """Load the EDA output for an experiment.

        Args:
            experiment_id: str or UUID identifier used to locate the artifact.

        Returns:
            A pandas DataFrame containing the persisted EDA data.
        """
        data = uf.read_df(experiment_id, name=constants.EDA_NAME)
        return data

    def _save_processed_data(self):
        """Persist each processed split as an experiment artifact."""
        uf.save_df(
            self.train,
            self.experiment_id,
            name=(
                f"{constants.DATA_PREPROCESSING_NAME}_{constants.TRAIN_NAME}"
            ),
        )
        uf.save_df(
            self.val,
            self.experiment_id,
            name=f"{constants.DATA_PREPROCESSING_NAME}_{constants.VAL_NAME}",
        )
        uf.save_df(
            self.test,
            self.experiment_id,
            name=f"{constants.DATA_PREPROCESSING_NAME}_{constants.TEST_NAME}",
        )

    def _split_data(self):
        """Partition data chronologically at the configured boundaries."""
        data = self.data
        data.index = pd.to_datetime(data.index)
        self.train = data[data.index < self.train_val_split_date]
        self.val = data[
            (data.index >= self.train_val_split_date)
            & (data.index < self.val_test_split_date)
        ]
        self.test = data[data.index >= self.val_test_split_date]

    def _treat_missing_values(self):
        """Apply the configured missing-value treatment to all data splits."""
        features = self.train.drop(
            columns=[experiment_constants.TARGET_COL]
        ).columns
        if self.imputer in ["backfill", "bfill", "ffill"]:
            self.train.loc[:, features] = self.train[features].fillna(
                method=self.imputer
            )
            if len(self.val) > 0:
                self.val.loc[:, features] = self.val[features].fillna(
                    method=self.imputer
                )
            if len(self.test) > 0:
                self.test.loc[:, features] = self.test[features].fillna(
                    method=self.imputer
                )
        else:
            self.imputer.fit(self.train)
            self.train = pd.DataFrame(
                data=self.imputer.transform(self.train),
                index=self.train.index,
                columns=self.train.columns,
            )
            if len(self.val) > 0:
                self.val = pd.DataFrame(
                    data=self.imputer.transform(self.val),
                    index=self.val.index,
                    columns=self.val.columns,
                )
            if len(self.test) > 0:
                self.test = pd.DataFrame(
                    data=self.imputer.transform(self.test),
                    index=self.test.index,
                    columns=self.test.columns,
                )

    def execute_preprocessing(self):
        """Run preprocessing, split the data, and persist each result.

        The experiment callback runs before the chronological split so it can
        perform shared filtering, transformations, or feature generation.

        Returns:
            A tuple of pandas DataFrames ``(train, val, test)`` containing the
            processed splits.
        """
        self.data = self.preprocesing_function(self.trial, self.data)
        self._split_data()
        self._treat_missing_values()
        self._save_processed_data()
        return self.train, self.val, self.test
