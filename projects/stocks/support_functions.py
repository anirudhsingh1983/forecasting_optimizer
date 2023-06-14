import datetime
import pickle
import warnings
from functools import reduce

import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.cloud import bigquery_storage

from util import utility_functions as uf

pd.set_option('display.max_columns', None)

warnings.filterwarnings("ignore")

_bqclient = bigquery.Client(project='wf-gcp-us-ae-dsservice-prod')
_bqstorageclient = bigquery_storage.BigQueryReadClient()

value_key = 'Open'

def get_raw_data():
    stock = pd.read_csv("tmp/stocks/c3ai.csv")
    nasdaq = pd.read_csv("tmp/stocks/nasdaq.csv")

    nasdaq = nasdaq.set_index(keys=['Date'])[[value_key]].rename(columns={value_key: f"nasdaq_{value_key}"})
    stock = stock.set_index(keys=['Date'])[[value_key]].rename(columns={value_key: f"stock_{value_key}"})

    data = pd.concat([stock, nasdaq], axis=1).dropna().sort_index()
    return data


def preprocessing(trial, df_raw):
    return df_raw

def feature_engineering(trial, df):
    df = uf.get_date_features(trial, df, date_col=None)
    df = uf.get_lags(
        trial,
        df,
        cols=[f"stock_{value_key}", f"nasdaq_{value_key}"],
        lags={f"stock_{value_key}": [1,2,3], f"nasdaq_{value_key}": [1,2]},
        dropna=True,
    )
    df = df.drop(columns=[f"nasdaq_{value_key}"])
    return df