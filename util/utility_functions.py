import logging
import math
import pickle

import numpy as np
import pandas as pd
# import tensorflow as tf
from scipy.stats import nbinom
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

import constants


def identify_isolation_forest_outliers(data):
    clf = IsolationForest(n_estimators=100, warm_start=True)
    clf.fit(data)

    y_pred = clf.predict(data)
    y_pred = pd.Series(y_pred)
    if (-1 in y_pred.values):
        ol_count = y_pred.value_counts()[-1]  # -1 indicates outlier detected by isolation forest
    else:
        ol_count = 0
    logging.warning(f"The total number of outliers are: {ol_count}")
    mask = (y_pred == -1).values
    data_ol = data.loc[mask, :]
    data_norm = data.loc[~mask, :]
    return data_norm, data_ol, mask


def nested_dict_to_flatten_df(dc, levels, value_col_name='values'):
    """
    Get a flattened dataframe from a nested dictionary.
    """
    top_level = levels[0]
    keys = list(dc.keys())
    temp_dfs = list()
    if len(levels) > 1:
        for key in keys:
            remaining_levels = levels[1:]
            temp_df = nested_dict_to_flatten_df(dc=dc[key], levels=remaining_levels, value_col_name=value_col_name)
            temp_df[top_level] = key
            temp_dfs.append(temp_df)
        df = pd.concat(temp_dfs, axis=0)
    else:
        df = pd.Series(dc).to_frame().reset_index()
        df.columns = [top_level, value_col_name]
    df = df[levels + [value_col_name]]
    return df


def save_df(data, experiment_id, name):
    file_name = f"{constants.OUTPUT_DATA_FOLDER}/{experiment_id}_{name}.hdf"
    data.to_hdf(file_name, key=name)


def read_df(experiment_id, name):
    file_name = f"{constants.OUTPUT_DATA_FOLDER}/{experiment_id}_{name}.hdf"
    data = pd.read_hdf(file_name, key=name)
    return data


def save_data(data, experiment_id, name, name_len_limit=100):
    file_name = f"{constants.OUTPUT_DATA_FOLDER}/{experiment_id}_{name[:name_len_limit]}.pickle"
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def read_data(experiment_id, name):
    file_name = f"{constants.OUTPUT_DATA_FOLDER}/{experiment_id}_{name}.pickle"
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def getBoxplotOuliters(s: pd.Series):
    # get 25 and 75 percentiles
    p25 = s.quantile(0.25)
    p75 = s.quantile(0.75)
    # inter-quantile range
    iqr = p75 - p25

    # upper and lower limits
    upper_limit = p75 + 1.5 * iqr
    lower_limit = p25 - 1.5 * iqr

    return lower_limit, upper_limit


def deepsum(obj, exclude_dict_keys=[], ignore_na=True):
    if type(obj) in [int, float, np.int, np.int32, np.int64, np.float, np.float32, np.float64]:
        return obj

    sum = 0
    if type(obj) == dict:
        for key, value in obj.items():
            if key in exclude_dict_keys:
                continue
            val = deepsum(obj=value, exclude_dict_keys=exclude_dict_keys, ignore_na=ignore_na)
            if (val in [None, np.nan]) | (
                    np.isnan(
                        val)):  # check for [None, np.nan] is redundant but is kept to accomodate any package changes later
                if ignore_na:
                    continue
                else:
                    sum += val
            else:
                sum += val

    if type(obj) in [list, np.array, tuple]:
        for value in obj:
            val = deepsum(obj=value, exclude_dict_keys=exclude_dict_keys, ignore_na=ignore_na)
            if (val in [None, np.nan]) | (
                    np.isnan(
                        val)):  # check for [None, np.nan] is redundant but is kept to accomodate any package changes later
                if ignore_na:
                    continue
                else:
                    sum += val
            else:
                sum += val

    return sum


def deepsum_with_len(obj, exclude_dict_keys=[], ignore_na=True):
    if type(obj) in [int, float, np.int, np.int32, np.int64, np.float, np.float32, np.float64]:
        return obj, 1

    sum = 0
    len = 0
    if type(obj) == dict:
        for key, value in obj.items():
            if key in exclude_dict_keys:
                continue
            val, l = deepsum_with_len(obj=value, exclude_dict_keys=exclude_dict_keys, ignore_na=ignore_na)
            if (val in [None, np.nan]) | (
                    np.isnan(
                        val)):  # check for [None, np.nan] is redundant but is kept to accomodate any package changes later
                if ignore_na:
                    continue
                else:
                    len += l
                    sum += val
            else:
                len += l
                sum += val

    if type(obj) in [list, np.array, tuple]:
        for value in obj:
            val, l = deepsum_with_len(obj=value, exclude_dict_keys=exclude_dict_keys, ignore_na=ignore_na)
            if (val in [None, np.nan]) | (
                    np.isnan(
                        val)):  # check for [None, np.nan] is redundant but is kept to accomodate any package changes later
                if ignore_na:
                    continue
                else:
                    len += l
                    sum += val
            else:
                len += l
                sum += val

    return sum, len


def get_date_features(trial, df, date_col=None):
    """
    Get features from a specific date and add those features to the dataframe df.
    """
    if date_col is None:
        dates = df.index
        dates = pd.to_datetime(dates)  # to make sure the index is in datetime format
    else:
        dates = df[date_col]
        dates = pd.to_datetime(dates).dt

    df['day'] = dates.day
    df['dayofweek'] = dates.dayofweek
    df['week'] = dates.week
    df['month'] = dates.month

    return df


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def treat_categorical_features(encoder, train, val, test, categorical_features=None):
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown='ignore', min_frequency=1)

    if categorical_features is None:
        categorical_features = train.select_dtypes(exclude=np.number).columns.tolist()

    categorical_features = list(set(categorical_features))  # to remove duplicate features
    encoder.fit(train[categorical_features])
    encoded_columns = [x if (y is None) else ([e for e in x if e not in y] + ['infrequent_val']) for x, y in
                       tuple(zip(encoder.categories_, encoder.infrequent_categories_))]
    encoded_columns = dict(zip(categorical_features, encoded_columns))
    encoded_columns = [[f"{col}_{val}" for val in list(encoded_values)] for col, encoded_values in
                       encoded_columns.items()]
    encoded_columns = np.concatenate(encoded_columns)

    # encode dataframes
    train_encoded = pd.DataFrame(
        data=encoder.transform(train[categorical_features]).toarray(),
        index=train.index,
        columns=encoded_columns,
    )
    if val is not None:
        val_encoded = pd.DataFrame(
            data=encoder.transform(val[categorical_features]).toarray(),
            index=val.index,
            columns=encoded_columns,
        )
    if test is not None:
        test_encoded = pd.DataFrame(
            data=encoder.transform(test[categorical_features]).toarray(),
            index=test.index,
            columns=encoded_columns,
        )

    # join with remaining columns
    train = pd.concat([train.drop(columns=categorical_features), train_encoded], axis=1)
    if val is not None:
        val = pd.concat([val.drop(columns=categorical_features), val_encoded], axis=1)
    if test is not None:
        test = pd.concat([test.drop(columns=categorical_features), test_encoded], axis=1)

    return encoder, encoded_columns, train, val, test


def get_week_from_date(df, date_col):
    """
    Create a new column that has week starting date in it.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    var = df[date_col].dt.dayofweek + 1
    var = var.mask(var == 7, 0)
    df['week'] = df[date_col] - pd.to_timedelta(var, unit='days')
    df['week'] = df['week'].dt.date
    return df


def deepbrowse(obj, func, exclude_dict_keys=[]):
    if type(obj) in [int, float, np.int, np.int32, np.int64, np.float, np.float32, np.float64]:
        func(obj)

    if type(obj) == dict:
        for key, value in obj.items():
            if key in exclude_dict_keys:
                continue
            deepbrowse(obj=value, func=func, exclude_dict_keys=exclude_dict_keys)

    if type(obj) in [list, np.array, tuple]:
        for value in obj:
            deepbrowse(obj=value, func=func, exclude_dict_keys=exclude_dict_keys)


def kl_divergence(actual_probabilities, predicted_probabilities, base=2):
    e = 0
    for pp, ap in zip(predicted_probabilities, actual_probabilities):
        e = e + ap * (math.log(pp / ap, base))

    return (-e)


def js_divergence(actual_density, pred_density, version='auf', base=2):
    """
    Js divergence as per custom KL divergence function or using tensorflow based on the value of argument 'version'.
    This uses the standard definition of JS divergence instead of using the sq. root version used in JS distance.
    """
    actual_density = np.array(actual_density)
    pred_density = np.array(pred_density)
    m = (actual_density + pred_density) / 2
    if version == 'auf':
        p = kl_divergence(actual_density, m, base=base)
        q = kl_divergence(pred_density, m, base=base)
    elif version == 'tf':
        kl = tf.keras.losses.KLDivergence()
        p = kl(list(actual_density), list(m)).numpy()
        q = kl(list(pred_density), list(m)).numpy()

    jsd = (p + q) / 2
    return jsd


def get_templated_distribution(df, col, prob_template):
    vc = df[col].value_counts(normalize=True)
    vc = vc + prob_template
    vc = vc.fillna(0)[prob_template.index]
    return vc


def get_jsd(s, classes):
    vc = pd.Series(s[constants.ACTUAL_DISTRIBUTION_STR])
    prob_template = pd.Series(data=0, index=classes)
    vc = prob_template + vc
    vc = vc.fillna(0)
    probs = pd.concat([vc, s[classes]], axis=1).fillna(0)
    jsd = js_divergence(actual_density=probs.iloc[:, 0], pred_density=probs.iloc[:, 1], version='tf', base=2)
    return jsd


def get_nb_dist(mean, std):
    p = mean / (std ** 2)
    n = (mean ** 2) / (std ** 2 - mean)
    dist = nbinom(n, p)
    return dist


def get_nb_probs(dist, mean, std, num_std_spread):
    ll = math.floor((mean - num_std_spread * std))
    if ll < 0:
        ll = 0
    ul = math.ceil((mean + num_std_spread * std))
    if ul < 0:
        ul = 0
    range = np.arange(ll, ul + 1)
    probs = dist.pmf(range)
    probs = probs / probs.sum()
    probs = pd.Series(data=probs, index=range)
    return probs


def get_norm_probs(dist, mean, std, num_std_spread):
    ll = math.floor((mean - num_std_spread * std))
    if ll < 0:
        ll = 0
    ul = math.ceil((mean + num_std_spread * std))
    if ul < 0:
        ul = 0
    range = np.arange(ll, ul + 1)
    probs = dist.pdf(range)
    probs = probs / probs.sum()
    probs = pd.Series(data=probs, index=range)
    return probs


def convert_date_to_week(s, index=True, date_col=None):
    if index:
        days = pd.to_datetime(pd.Series(s.index))
        s.index = days.dt.to_period('W').apply(lambda r: r.start_time)
    else:
        if isinstance(s, pd.Series):
            days = pd.to_datetime(s)
            s = days.dt.to_period('W').apply(lambda r: r.start_time)
        elif isinstance(s, pd.DataFrame):
            days = pd.to_datetime(s[date_col])
            s[date_col] = days.dt.to_period('W').apply(lambda r: r.start_time)
        else:
            logging.error(f"argument s is not a valid object type.")
    return s


def cs_to_seq(x, y, length, sampling_rate, stride):
    data_gen = TimeseriesGenerator(x, y, length=length, sampling_rate=sampling_rate, batch_size=1, stride=stride)
    x = np.array([seq for seq, _ in data_gen]).squeeze()
    y = np.array([target for _, target in data_gen])
    return x, y


def get_lags(trial, df, cols, lags=[1], dropna=True):
    for col in cols:
        if isinstance(lags, dict):
            col_lags = lags[col]
        else:
            col_lags = lags

        for lag in col_lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    if dropna:
        if isinstance(lags, dict):
            max_lag = np.concatenate(list(lags.values())).max()
        else:
            max_lag = np.array(lags).max()

        df = df.iloc[max_lag:, :]

    return  df
