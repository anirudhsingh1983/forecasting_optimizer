import pickle
import warnings
from functools import reduce
from typing import List, Dict
import datetime

import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.cloud import bigquery_storage
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from tqdm import tqdm

from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit as ts_split

pd.set_option('display.max_columns', None)

warnings.filterwarnings("ignore")

_bqclient = bigquery.Client(project='wf-gcp-us-ae-dsservice-prod')
_bqstorageclient = bigquery_storage.BigQueryReadClient()

y_var = "presentedVolume"
x_var = ["exp_mov_avg",
         "exp_ma_orders",
         "tot_na_fcst",
         "exp_cnt_orders_lag1",
         "holiday_wf",
         "memo_day",
         "bf_july",
         "july_4",
         "thanksgiving",
         "cyber_monday",
         "thanksgiving_christmas",
         "christmas_eve",
         "new_years_eve",
         "new_year",
         ]

backtesting_period = 53

coln = ["week", y_var]
coln.extend(x_var)

def est_current_week_val(df: pd.DataFrame,
                         var: str,
                         n: int = 3,
                         drop: bool = True) -> pd.DataFrame:
    """
        Gets estimate of a feature that is not available because the week has nto complete yet.
    """
    tmp_coln = ["week", "day", var]
    df = df[tmp_coln]
    df_tmp = df[df["day"] == 1]
    df_tmp_1 = df_tmp.groupby(by=["week"], axis=0)[var].sum().reset_index()
    df_tmp = df[df["day"] == 0]
    df_tmp_0 = df_tmp.groupby(by=["week"], axis=0)[var].sum().reset_index()
    df_tmp_op = pd.merge(df_tmp_1,
                         df_tmp_0,
                         on="week",
                         # how="inner")
                         how="outer").fillna(0)  # to avoid curtailing the last but partial week
    df_tmp_op.columns = ["week", "1", "0"]
    df_tmp_op["tot"] = df_tmp_op["0"] + df_tmp_op["1"]
    df_tmp_op["perc"] = df_tmp_op["0"] / (df_tmp_op["tot"])
    df_tmp_op["avg_perc"] = 0
    for i in range(1, n):
        df_tmp_op["avg_perc"] += df_tmp_op["perc"].shift(i)
    df_tmp_op["avg_perc"] /= n - 1
    df_tmp_op["est_var"] = np.round(df_tmp_op["1"] / (1 - df_tmp_op["avg_perc"]))
    if drop:
        df_tmp_op = df_tmp_op[["week", "tot", "est_var"]]
        df_tmp_op.columns = ["week", var, "est_" + str(var)]
    return df_tmp_op


def ignore_warnings(test_func):
    """Suppress warnings."""

    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)

    return do_test


def get_raw_data():
    local = True

    if local:
        with open('output_data/cf_raw_data.pickle', 'rb') as f:
            df_raw = pickle.load(f)

    else:
        try:
            dataQuery = "select * from wf-gcp-us-ae-dsservice-prod.junk.dailyContactsOrderHolidaysNaB2cV3  order by actualWeek, presentedDate;"
            result_stream = _bqclient.query(dataQuery).result()
            df_raw = result_stream.to_dataframe(bqstorage_client=_bqstorageclient)
        except:
            with open('projects/contact_forecasting/queries/historical_data.sql', 'r') as f:
                dataQuery = f.read()
            result_stream = _bqclient.query(dataQuery).result()
            df_raw = result_stream.to_dataframe(bqstorage_client=_bqstorageclient)

        df_raw.index = df_raw['presentedDate']
        df_raw = df_raw[df_raw['presentedDate'] <= pd.to_datetime("2022-10-01")]
        # df_raw = df_raw.drop(columns=['actualWeek', 'presentedDate'])

        with open('output_data/cf_raw_data.pickle', 'wb') as f:
            pickle.dump(df_raw, f)

    return df_raw

def preprocessing(trial, df_raw):
    df_raw = df_raw.copy()
    df_raw.columns = ["week", "date", "presentedVolume", "cnt_orders",
                      "perc_new_cust_orders", "holiday_us", "holiday_ca"]
    print(df_raw.tail())

    # NA Forecast
    naForecastQuery = "select * from wf-gcp-us-ae-dsservice-prod.driver_based_forecasting.tbl_dbf_features_frm_order_forecasts"
    result_stream = _bqclient.query(naForecastQuery).result()
    df_fcst = result_stream.to_dataframe(bqstorage_client=_bqstorageclient)
    df_fcst.sort_values(by=['forecastWeek']).tail()

    df_fcst = df_fcst.rename(columns={'forecastWeek': 'week', 'TotalOrderForecastNA_extra': 'tot_na_fcst'})
    df_fcst.sort_values(by=['week']).tail()

    # --- Data pre-processing
    df_raw["week"] = pd.to_datetime(df_raw["week"])
    df_raw["date"] = pd.to_datetime(df_raw["date"])

    df_fcst["week"] = pd.to_datetime(df_fcst["week"])

    df_raw["day"] = np.where((df_raw["date"].dt.dayofweek >= 3)
                             & (df_raw["date"].dt.dayofweek < 6), 0, 1)

    df_raw["new_cust"] = np.ceil(df_raw["cnt_orders"]
                                 * df_raw["perc_new_cust_orders"])

    # Add Wayday 2022
    df_raw["holiday_us"] = np.where((df_raw["date"] == "2022-04-27")
                                    | (df_raw["date"] == "2022-04-28"),
                                    "Wayday", df_raw["holiday_us"])

    # ---- Holiday flags
    df_raw["holiday_wf"] = np.where(df_raw["holiday_us"] == "Wayday", 1, 0)

    df_raw["memo_day"] = np.where(df_raw["holiday_us"] == "Memorial Day", 1, 0)

    df_raw["bf_july"] = np.where(df_raw["holiday_us"] == "Black Friday In July",
                                 1, 0)

    df_raw["july_4"] = np.where(df_raw["holiday_us"] == "Independence Day", 1, 0)

    df_raw["thanksgiving"] = np.where(df_raw["holiday_us"] == "Thanksgiving Day",
                                      1, 0)

    df_raw["christmas_eve"] = np.where(df_raw["holiday_us"] == "Christmas Eve",
                                       1, 0)

    df_raw["thanksgiving_christmas"] = df_raw["thanksgiving"] \
                                       + df_raw["christmas_eve"]

    df_raw["cyber_monday"] = np.where(df_raw["holiday_us"] == "Cyber Monday",
                                      1, 0)

    df_raw["new_years_eve"] = np.where(df_raw["holiday_us"] == "New Years Eve",
                                       1, 0)

    df_raw["new_year"] = np.where(df_raw["holiday_us"] == "New Years Day",
                                  1, 0)

    t_c = "thanksgiving_christmas"

    df_hol = reduce(lambda x, y: pd.merge(x, y, on="week", how="inner"),
                    [est_current_week_val(df_raw, "holiday_wf")[["week",
                                                                 "holiday_wf"]],
                     est_current_week_val(df_raw, "memo_day")[["week",
                                                               "memo_day"]],
                     est_current_week_val(df_raw, "bf_july")[["week",
                                                              "bf_july"]],
                     est_current_week_val(df_raw, "july_4")[["week",
                                                             "july_4"]],
                     est_current_week_val(df_raw,
                                          "thanksgiving")[["week",
                                                           "thanksgiving"]],
                     est_current_week_val(df_raw,
                                          "christmas_eve")[["week",
                                                            "christmas_eve"]],
                     est_current_week_val(df_raw, t_c)[["week", t_c]],
                     est_current_week_val(df_raw,
                                          "cyber_monday")[["week",
                                                           "cyber_monday"]],
                     est_current_week_val(df_raw,
                                          "new_years_eve")[["week",
                                                            "new_years_eve"]],
                     est_current_week_val(df_raw, "new_year")[["week",
                                                               "new_year"]]])

    df_raw.tail()

    nextWeek = pd.to_datetime(
        datetime.datetime.today() - pd.to_timedelta(datetime.datetime.today().weekday() + 1 - 7, unit='day')).date()
    # if nextWeek.strftime("%Y-%m-%d") != df_raw['week'].dt.strftime("%Y-%m-%d").iloc[-1]:
    #     cols = df_raw.drop(['week', 'date'], axis=1).columns
    #     newRow = {
    #         'week': pd.to_datetime(nextWeek),
    #         'date': pd.to_datetime(nextWeek),
    #     }
    #     newRow.update(dict(zip(cols, np.zeros(len(cols)))))
    #     df_raw = df_raw.append(newRow, ignore_index=True)

    holidayCols = ['holiday_wf',
                   'memo_day', 'bf_july', 'july_4', 'thanksgiving',
                   'christmas_eve', 'thanksgiving_christmas', 'cyber_monday', 'new_years_eve', 'new_year']
    df_hol = df_raw.groupby(by=['week'])[holidayCols].sum().reset_index()

    df_feat = reduce(lambda x, y: pd.merge(x, y, on="week", how="inner"),
                     [est_current_week_val(df_raw, "presentedVolume"),
                      df_fcst,
                      est_current_week_val(df_raw, "cnt_orders")])
    df_feat.tail()

    df_ip = pd.merge(df_feat,
                     df_hol,
                     on="week",
                     how="left")
    df_ip.tail()

    # Expected moving average
    df_ip["exp_mov_avg"] = df_ip["presentedVolume"].rolling(window=2).sum().shift(1) \
                           + df_ip["est_presentedVolume"]
    df_ip["exp_mov_avg"] = df_ip["exp_mov_avg"].shift(1) / 3

    # Expected moving average of orders
    df_ip["exp_ma_orders"] = df_ip["cnt_orders"].rolling(window=2).sum().shift(1) \
                             + df_ip["est_cnt_orders"]
    df_ip["exp_ma_orders"] = df_ip["exp_ma_orders"].shift(1) / 3

    # Expected call volume lag1
    df_ip["exp_tot_call_vol_lag1"] = df_ip["est_presentedVolume"].shift(1)

    # Expected count of orders
    df_ip["exp_cnt_orders_lag1"] = df_ip["est_cnt_orders"].shift(1)

    df_ipBkp = df_ip.copy()
    df_ip.tail()

    # Transformations
    df_ip = df_ipBkp.copy()
    df_ip["exp_mov_avg"] = df_ip["exp_mov_avg"] ** 1
    df_ip["exp_tot_call_vol_lag1"] = df_ip["exp_tot_call_vol_lag1"] ** 1
    df_ip["exp_ma_orders"] = df_ip["exp_ma_orders"] ** 1
    df_ip["exp_cnt_orders_lag1"] = df_ip["exp_cnt_orders_lag1"] ** 1
    df_ip["tot_na_fcst"] = df_ip["tot_na_fcst"] ** 3

    df_ip = df_ip[coln].dropna().reset_index(drop=True)
    df_ip.index = df_ip['week']
    df_ip = df_ip.drop(columns = ['week'])
    return df_ip


