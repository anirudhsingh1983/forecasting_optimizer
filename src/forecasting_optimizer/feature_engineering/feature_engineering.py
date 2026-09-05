# Copyright 2023 Anirudh Singh
# SPDX-License-Identifier: BSD-3-Clause

"""Generate, encode, and filter features for forecasting model inputs."""

import importlib
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from forecasting_optimizer import constants
from forecasting_optimizer import experiment_settings
from forecasting_optimizer import framework_settings
from forecasting_optimizer.feature_engineering import (
    feature_engineering_constants,
)
from forecasting_optimizer.util import utility_functions as uf

experiment_constants = importlib.import_module(
    "forecasting_optimizer.experiment_configs."
    f"{experiment_settings.EXPERIMENT_CONSTANTS_MODULE_NAME}"
)


class FeatureEngineering:
    """Build model features for train, validation, and test splits.

    Attributes:
        experiment_id: str or UUID identifier for experiment artifacts.
        trial: object representing the optimization trial passed to callbacks.
        data: tuple of source train, validation, and test DataFrames.
        outlier_method: Optional str naming the outlier-detection method.
        categorical_feature_encoder: Value expected to implement the encoder
            interface used by this class. The legacy ``None`` fallback is the
            default outlier-method string, not a usable encoder.
        feature_engineering_scaler: Value expected to implement the scaler
            interface used by the optional scaling step.
        feature_development_function: Callable used for feature creation.
        categorical_engineered_features: list of columns requiring encoding.
        train: pandas DataFrame populated during execution.
        val: pandas DataFrame populated during execution.
        test: pandas DataFrame populated during execution.
    """

    def __init__(
        self,
        experiment_id,
        trial,
        data=None,
        outlier_method=None,
        categorical_feature_encoder=None,
        feature_engineering_scaler=None,
    ):
        """Initialize feature engineering for an experiment.

        Args:
            experiment_id: str or UUID identifier used for artifacts.
            trial: object representing the optimization trial passed to the
                experiment callback.
            data: Optional tuple of train, validation, and test pandas
                DataFrames. If omitted, persisted artifacts are loaded.
            outlier_method: Optional str naming the outlier-detection method.
            categorical_feature_encoder: Encoder object. Passing ``None``
                retains the legacy default-outlier-method assignment, which
                does not implement the encoder interface used during execution.
            feature_engineering_scaler: Optional scaler object override.
        """
        self.experiment_id = experiment_id
        self.trial = trial
        if data is None:
            self.data = self._load_data(experiment_id=self.experiment_id)
        else:
            self.data = data

        self.outlier_method = outlier_method

        if categorical_feature_encoder is None:
            self.categorical_feature_encoder = (
                framework_settings.DEFAULT_OUTLIER_METHOD
            )
        else:
            self.categorical_feature_encoder = categorical_feature_encoder

        if feature_engineering_scaler is None:
            self.feature_engineering_scaler = (
                framework_settings.DEFAULT_FEATURE_ENGINEERING_SCALER
            )
        else:
            self.feature_engineering_scaler = feature_engineering_scaler

        # The trial lets experiments optimize feature-development choices.
        try:
            feature_development_function = (
                experiment_constants.FEATURE_ENGINEERING_FUNCTION
            )
            if feature_development_function is None:
                feature_development_function = lambda trial, x: x
        except:
            feature_development_function = lambda trial, x: x
        self.feature_development_function = feature_development_function

        try:
            categorical_engineered_features = (
                experiment_constants.CATEGORICAL_ENGINEERED_FEATURES
            )
            if categorical_engineered_features is None:
                categorical_engineered_features = []
        except:
            categorical_engineered_features = []
        self.categorical_engineered_features = categorical_engineered_features

    def _load_data(self, experiment_id):
        """Load all persisted preprocessing splits for an experiment.

        Args:
            experiment_id: str or UUID identifier used to locate artifacts.

        Returns:
            A tuple of pandas DataFrames ``(train, val, test)`` containing the
            persisted splits.
        """
        train = uf.read_df(
            experiment_id,
            name=f"{constants.DATA_PREPROCESSING_NAME}_{constants.TRAIN_NAME}",
        )
        val = uf.read_df(
            experiment_id,
            name=f"{constants.DATA_PREPROCESSING_NAME}_{constants.VAL_NAME}",
        )
        test = uf.read_df(
            experiment_id,
            name=f"{constants.DATA_PREPROCESSING_NAME}_{constants.TEST_NAME}",
        )
        return train, val, test

    def _save_processed_data(self):
        """Persist each engineered split as an experiment artifact."""
        uf.save_df(
            self.train,
            self.experiment_id,
            name=f"{constants.FEATURE_ENGINEERING_NAME}_{constants.TRAIN_NAME}",
        )
        uf.save_df(
            self.val,
            self.experiment_id,
            name=f"{constants.FEATURE_ENGINEERING_NAME}_{constants.VAL_NAME}",
        )
        uf.save_df(
            self.test,
            self.experiment_id,
            name=f"{constants.FEATURE_ENGINEERING_NAME}_{constants.TEST_NAME}",
        )

    def _generate_features(self):
        """Apply the experiment feature callback independently to each split."""
        train, val, test = self.data
        self.train = self.feature_development_function(self.trial, train)
        self.val = self.feature_development_function(self.trial, val)
        self.test = self.feature_development_function(self.trial, test)

    def _treat_categorical_features(self):
        """Fit the encoder on training data and encode each split."""
        train, val, test = self.train, self.val, self.test
        encoder = self.categorical_feature_encoder
        categorical_features = (
            train.select_dtypes(include=constants.OBJECT_NAME).columns.tolist()
            + self.categorical_engineered_features
        )
        # Set intersection removes duplicate configured and inferred features.
        categorical_features = list(
            set(categorical_features).intersection(train.columns)
        )
        encoder.fit(train[categorical_features])
        encoded_columns = [
            (
                x
                if (y is None)
                else ([e for e in x if e not in y] + ["infrequent_val"])
            )
            for x, y in tuple(
                zip(encoder.categories_, encoder.infrequent_categories_)
            )
        ]
        encoded_columns = dict(zip(categorical_features, encoded_columns))
        encoded_columns = [
            [f"{col}_{val}" for val in list(encoded_values)]
            for col, encoded_values in encoded_columns.items()
        ]
        if len(encoded_columns) > 0:
            encoded_columns = np.concatenate(encoded_columns)

        train_encoded = pd.DataFrame(
            data=encoder.transform(train[categorical_features]).toarray(),
            index=train.index,
            columns=encoded_columns,
        )

        if len(self.val) > 0:
            val_encoded = pd.DataFrame(
                data=encoder.transform(val[categorical_features]).toarray(),
                index=val.index,
                columns=encoded_columns,
            )
        else:
            val_encoded = pd.DataFrame(columns=train_encoded.columns)

        if len(self.test) > 0:
            test_encoded = pd.DataFrame(
                data=encoder.transform(test[categorical_features]).toarray(),
                index=test.index,
                columns=encoded_columns,
            )
        else:
            test_encoded = pd.DataFrame(columns=train_encoded.columns)

        self.train = pd.concat(
            [train.drop(columns=categorical_features), train_encoded], axis=1
        )
        self.val = pd.concat(
            [val.drop(columns=categorical_features), val_encoded], axis=1
        )
        self.test = pd.concat(
            [test.drop(columns=categorical_features), test_encoded], axis=1
        )

    def _scale_data(self):
        """Fit the configured scaler on training data and scale each split."""
        scaler = self.feature_engineering_scaler
        scaler.fit(self.train)
        self.train = pd.DataFrame(
            data=scaler.transform(self.train),
            index=self.train.index,
            columns=self.train.columns,
        )
        if len(self.val) > 0:
            self.val = pd.DataFrame(
                data=scaler.transform(self.val),
                index=self.val.index,
                columns=self.val.columns,
            )
        if len(self.test) > 0:
            self.test = pd.DataFrame(
                data=scaler.transform(self.test),
                index=self.test.index,
                columns=self.test.columns,
            )

    def _identify_isolation_forest_outliers(self, data, clf, data_name="data"):
        """Partition data according to predictions from a fitted detector.

        Args:
            data: pandas DataFrame to partition.
            clf: Fitted estimator exposing an Isolation Forest-style
                ``predict`` method.
            data_name: str split name included in the warning message.

        Returns:
            A tuple of pandas DataFrames ``(normal_data, outlier_data)``
            partitioned by predictions.
        """
        y_pred = clf.predict(data)
        y_pred = pd.Series(y_pred)
        if -1 in y_pred.values:
            # Isolation Forest uses -1 to label detected outliers.
            ol_count = y_pred.value_counts()[-1]
        else:
            ol_count = 0
        logging.warning(
            f"The total number of outliers in {data_name} are: {ol_count}"
        )
        mask = (y_pred == -1).values
        data_ol = data.loc[mask, :]
        data_norm = data.loc[~mask, :]
        return data_norm, data_ol

    def _remove_outliers(self):
        """Remove training outliers when the configured method permits it.

        Only Isolation Forest removal is implemented. Detected outliers are
        retained when removing them would discard at least 30% of training data
        or leave too few rows for sequential modeling.
        """
        try:
            seq_data_len = experiment_constants.SEQ_DATA_LEN
        except:
            seq_data_len = framework_settings.DEFAULT_SEQ_DATA_LEN

        try:
            seq_sampling = experiment_constants.SEQ_SAMPLING
        except:
            seq_sampling = framework_settings.DEFAULT_SEQ_SAMPLING

        if self.outlier_method is None:
            pass
        elif self.outlier_method == constants.OUTLIER_METHOD_BOXPLOT_NAME:
            pass
        elif self.outlier_method == constants.OUTLIER_METHOD_ZSCORE_NAME:
            pass
        elif self.outlier_method == constants.OUTLIER_METHOD_SVMONECLASS_NAME:
            pass
        elif (
            self.outlier_method == constants.OUTLIER_METHOD_ISOLATIONFOREST_NAME
        ):
            clf = IsolationForest(n_estimators=100, warm_start=True)
            clf.fit(self.train)
            data_norm, data_ol = self._identify_isolation_forest_outliers(
                self.train, clf, data_name="train"
            )
            if (len(data_ol) < (0.3 * len(self.train))) & (
                len(data_norm) > (seq_data_len + seq_sampling)
            ):
                self.train = data_norm

    def execute_feature_engineering(self):
        """Generate, encode, filter, and persist feature data.

        Returns:
            A tuple of pandas DataFrames ``(train, val, test)`` containing the
            engineered splits.
        """
        self._generate_features()
        self._treat_categorical_features()
        self._remove_outliers()
        self._save_processed_data()
        return self.train, self.val, self.test
