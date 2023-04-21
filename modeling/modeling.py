import importlib
import logging
from datetime import datetime

import numpy as np

import constants
import experiment_settings
import framework_settings
from modeling import modeling_constants
from modeling.model_classes import MODEL_CLASSES
from util import utility_functions as uf

experiment_constants = importlib.import_module(
    f"experiment_configs.{experiment_settings.EXPERIMENT_CONSTANTS_MODULE_NAME}")

class Modeling():
    def __init__(self, experiment_id, trial, data=None, model=None):
        self.experiment_id = experiment_id
        self.trial = trial

        if data is None:
            self.data = self._load_data(experiment_id=self.experiment_id)
        else:
            self.data = data

        if model is None:
            self.model = framework_settings.DEFAULT_MODEL_TO_TRAIN
        else:
            self.model = model

        self._models_to_train = [self.model]

    def _load_data(self):
        train = uf.read_df(self.experiment_id, name=f"{constants.FEATURE_ENGINEERING_NAME}_{constants.TRAIN_NAME}")
        val = uf.read_df(self.experiment_id, name=f"{constants.FEATURE_ENGINEERING_NAME}_{constants.VAL_NAME}")
        test = uf.read_df(self.experiment_id, name=f"{constants.FEATURE_ENGINEERING_NAME}_{constants.TEST_NAME}")
        return train, val, test

    def _train_one_model(self, model_name):
        _model_class = MODEL_CLASSES[model_name]
        model = _model_class(data=self.data, target_col_name=experiment_constants.TARGET_COL, experiment_id=self.experiment_id)
        model, best_params = model.fit()
        train_metric, val_metric, test_metric = model.evaluate()
        logging.info(
            f"Training performance: {train_metric} \n Val performance: {val_metric} \n Test performance: {test_metric}")
        return train_metric, val_metric, test_metric, best_params

    def execute_modeling(self):
        train_metric, val_metric, test_metric, best_params = self._train_one_model(
            model_name=self.model,
        )
        model_performance = {
            constants.BEST_PARAMETERS_NAME: best_params,
            constants.TRAIN_NAME: train_metric,
            constants.VAL_NAME: val_metric,
            constants.TEST_NAME: test_metric,
        }
        model_key = ""
        for k, v in model_performance[constants.BEST_PARAMETERS_NAME].items():
            if isinstance(v, float):
                v = np.round(v, 4)
            model_key = f"{model_key}_{k}_{v}"

        uf.save_data(model_performance, self.experiment_id,
                     name=f"{constants.MODELING_NAME}_{model_key}_{constants.MODEL_PERFORMANCE_RESULT_NAME}")
        return model_performance
