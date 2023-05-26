import importlib
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

import constants
import experiment_settings
import framework_settings
from feature_engineering import feature_engineering_constants
from util import utility_functions as uf

experiment_constants = importlib.import_module(
    f"experiment_configs.{experiment_settings.EXPERIMENT_CONSTANTS_MODULE_NAME}")


class FeatureEngineering():
    def __init__(
            self,
            experiment_id,
            trial,
            data=None,
            outlier_method=None,
            categorical_feature_encoder=None,
            feature_engineering_scaler=None
    ):
        self.experiment_id = experiment_id
        self.trial = trial
        if data is None:
            self.data = self._load_data(experiment_id=self.experiment_id)
        else:
            self.data = data

        self.outlier_method = outlier_method

        if categorical_feature_encoder is None:
            self.categorical_feature_encoder = framework_settings.DEFAULT_OUTLIER_METHOD
        else:
            self.categorical_feature_encoder = categorical_feature_encoder

        if feature_engineering_scaler is None:
            self.feature_engineering_scaler = framework_settings.DEFAULT_FEATURE_ENGINEERING_SCALER
        else:
            self.feature_engineering_scaler = feature_engineering_scaler

        # Note: feature_development_function will have a mandatory argument 'trial', so it can have optimizable arguments to leverage various data preprocessing ways to get the best outcome
        try:
            feature_development_function = experiment_constants.FEATURE_ENGINEERING_FUNCTION
            if feature_development_function is None:
                feature_development_function = lambda trial, x: x
        except:
            feature_development_function = lambda trial, x: x
        self.feature_development_function = feature_development_function

        try:
            categorical_engineered_features = experiment_constants.CATEGORICAL_ENGINEERED_FEATURES
            if categorical_engineered_features is None:
                categorical_engineered_features = []
        except:
            categorical_engineered_features = []
        self.categorical_engineered_features = categorical_engineered_features


    def _load_data(self, experiment_id):
        train = uf.read_df(experiment_id, name=f"{constants.DATA_PREPROCESSING_NAME}_{constants.TRAIN_NAME}")
        val = uf.read_df(experiment_id, name=f"{constants.DATA_PREPROCESSING_NAME}_{constants.VAL_NAME}")
        test = uf.read_df(experiment_id, name=f"{constants.DATA_PREPROCESSING_NAME}_{constants.TEST_NAME}")
        return train, val, test


    def _save_processed_data(self):
        uf.save_df(self.train, self.experiment_id, name=f"{constants.FEATURE_ENGINEERING_NAME}_{constants.TRAIN_NAME}")
        uf.save_df(self.val, self.experiment_id, name=f"{constants.FEATURE_ENGINEERING_NAME}_{constants.VAL_NAME}")
        uf.save_df(self.test, self.experiment_id, name=f"{constants.FEATURE_ENGINEERING_NAME}_{constants.TEST_NAME}")


    def _generate_features(self):
        train, val, test = self.data
        self.train = self.feature_development_function(self.trial, train)
        self.val = self.feature_development_function(self.trial, val)
        self.test = self.feature_development_function(self.trial, test)

    def _treat_categorical_features(self):
        train, val, test = self.train, self.val, self.test
        encoder = self.categorical_feature_encoder
        categorical_features = train.select_dtypes(
            include=constants.OBJECT_NAME).columns.tolist() + self.categorical_engineered_features
        categorical_features = list(set(categorical_features).intersection(train.columns))  # to remove duplicate features
        encoder.fit(train[categorical_features])
        encoded_columns = [x if (y is None) else ([e for e in x if e not in y] + ['infrequent_val']) for x, y in
                           tuple(zip(encoder.categories_, encoder.infrequent_categories_))]
        encoded_columns = dict(zip(categorical_features, encoded_columns))
        encoded_columns = [[f"{col}_{val}" for val in list(encoded_values)] for col, encoded_values in
                           encoded_columns.items()]
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

        self.train = pd.concat([train.drop(columns=categorical_features), train_encoded], axis=1)
        self.val = pd.concat([val.drop(columns=categorical_features), val_encoded], axis=1)
        self.test = pd.concat([test.drop(columns=categorical_features), test_encoded], axis=1)


    def _scale_data(self):
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

    def _identify_isolation_forest_outliers(self, data, clf, data_name='data'):
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


    def _remove_outliers(self):
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
        elif self.outlier_method == constants.OUTLIER_METHOD_ISOLATIONFOREST_NAME:
            clf = IsolationForest(n_estimators=100, warm_start=True)
            clf.fit(self.train)
            data_norm, data_ol = self._identify_isolation_forest_outliers(self.train, clf, data_name='train')
            if ((len(data_ol) < (0.3*len(self.train))) & (len(data_norm) > (seq_data_len+seq_sampling))):
                self.train = data_norm
            # self.val, _ = self._identify_isolation_forest_outliers(self.val, clf, data_name='val')
            # self.test, _ = self._identify_isolation_forest_outliers(self.test, clf, data_name='test')


    def execute_feature_engineering(self):
        self._generate_features()
        self._treat_categorical_features()
        # self._scale_data()
        self._remove_outliers()
        self._save_processed_data()
        return self.train, self.val, self.test
