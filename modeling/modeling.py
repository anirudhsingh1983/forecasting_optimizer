import importlib
import logging
from datetime import datetime

import constants
import experiment_settings
import framework_settings
from modeling import modeling_constants
from modeling.model_classes import MODEL_CLASSES
from util import utility_functions as uf

experiment_constants = importlib.import_module(
    f"experiment_configs.{experiment_settings.EXPERIMENT_CONSTANTS_MODULE_NAME}")

try:
    _models_to_train = experiment_constants.MODELS_TO_TRAIN
    if _models_to_train is None:
        _models_to_train = [framework_settings.DEFAULT_MODEL_TO_TRAIN]
    elif _models_to_train == 'all':
        _models_to_train = modeling_constants.MODEL_NAMES
    elif isinstance(_models_to_train, (list, tuple)):
        for model in _models_to_train:
            if model not in modeling_constants.MODEL_NAMES:
                raise Exception(f"invalid model name {model}")  # to be changed while adding exception handling properly
    else:
        raise Exception("invalid model name(s)")  # to be changed while adding exception handling properly
except:
    _models_to_train = [framework_settings.DEFAULT_MODEL_TO_TRAIN]

start = datetime.now()
log = logging.getLogger(__name__)


def _load_data(experiment_id):
    train = uf.read_df(experiment_id, name=f"{constants.FEATURE_ENGINEERING_NAME}_{constants.TRAIN_NAME}")
    val = uf.read_df(experiment_id, name=f"{constants.FEATURE_ENGINEERING_NAME}_{constants.VAL_NAME}")
    test = uf.read_df(experiment_id, name=f"{constants.FEATURE_ENGINEERING_NAME}_{constants.TEST_NAME}")
    return train, val, test


def _save_model_outputs(data, experiment_id):
    train, val, test = data
    uf.save_df(train, experiment_id, name=f"{constants.MODELING_NAME}_{constants.TRAIN_NAME}")
    uf.save_df(val, experiment_id, name=f"{constants.MODELING_NAME}_{constants.VAL_NAME}")
    uf.save_df(test, experiment_id, name=f"{constants.MODELING_NAME}_{constants.TEST_NAME}")


def _train_one_model(model_name, data, experiment_id):
    _model_class = MODEL_CLASSES[model_name]
    model = _model_class(data=data, target_col_name=experiment_constants.TARGET_COL, experiment_id=experiment_id)
    model, best_params = model.fit()
    train_metric, val_metric, test_metric = model.evaluate()
    log.info(
        f"Training performance: {train_metric} \n Val performance: {val_metric} \n Test performance: {test_metric}")
    return train_metric, val_metric, test_metric, best_params


def execute_modeling(experiment_id):
    train, val, test = _load_data(experiment_id=experiment_id)
    model_performances = dict()
    for model_name in _models_to_train:
        train_metric, val_metric, test_metric, best_params = _train_one_model(
            model_name=model_name,
            data=(train, val, test),
            experiment_id=experiment_id,
        )
        model_performances[model_name] = {
            constants.BEST_PARAMETERS_NAME: best_params,
            constants.TRAIN_NAME: train_metric,
            constants.VAL_NAME: val_metric,
            constants.TEST_NAME: test_metric,
        }

    uf.save_data(model_performances, experiment_id,
                 name=f"{constants.MODELING_NAME}_{constants.MODEL_PERFORMANCE_RESULT_NAME}")
    return model_performances
