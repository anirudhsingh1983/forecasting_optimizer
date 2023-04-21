import constants
from util import utility_functions as uf

def execute_data_landing(experiment_id, data_loading_function):
    data = data_loading_function()
    uf.save_df(data, experiment_id, name=constants.LANDING_DATA_NAME)
    return data
