import datetime
import pickle
import warnings
from functools import reduce

import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.cloud import bigquery_storage
import yfinance as yf

from util import utility_functions as uf

pd.set_option('display.max_columns', None)

warnings.filterwarnings("ignore")

_bqclient = bigquery.Client(project='wf-gcp-us-ae-dsservice-prod')
_bqstorageclient = bigquery_storage.BigQueryReadClient()

value_key = 'Open'
ticker = 'PLTR'
local = True
start='2015-01-01'

def get_raw_data():
    if local:
        stock = pd.read_csv(f"tmp/stocks/{ticker}.csv")
        nasdaq = pd.read_csv("tmp/stocks/nasdaq.csv")
        stock['Date'] = pd.to_datetime(stock['Date'], errors='coerce', format='%Y-%m-%d %H:%M:%S', utc=True).dt.date
        nasdaq['Date'] = pd.to_datetime(nasdaq['Date'], errors='coerce', format='%Y-%m-%d %H:%M:%S', utc=True).dt.date
        nasdaq = nasdaq.set_index(keys=['Date'])
        stock = stock.set_index(keys=['Date'])
    else:
        stock = yf.Ticker(ticker).history(start=start)
        nasdaq = yf.Ticker('^IXIC').history(start=start)
        stock.to_csv(f"tmp/stocks/{ticker}.csv")
        nasdaq.to_csv(f"tmp/stocks/nasdaq.csv")
        stock.index = pd.to_datetime(stock.index, errors='coerce', format='%Y-%m-%d %H:%M:%S', utc=True).date
        nasdaq.index = pd.to_datetime(nasdaq.index, errors='coerce', format='%Y-%m-%d %H:%M:%S', utc=True).date

    nasdaq = nasdaq[[value_key]].rename(columns={value_key: f"nasdaq_{value_key}"})
    stock = stock[[value_key]].rename(columns={value_key: f"stock_{value_key}"})

    data = pd.concat([stock, nasdaq], axis=1).dropna().sort_index()
    data = data.diff(periods=1).iloc[1:]
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
    df = uf.get_diffs(
        trial,
        df,
        cols=[col for col in df.columns if 'lag' in col],
        diffs=[1,2],
        dropna=True,
    )
    df = df.drop(columns=[f"nasdaq_{value_key}"])
    return df