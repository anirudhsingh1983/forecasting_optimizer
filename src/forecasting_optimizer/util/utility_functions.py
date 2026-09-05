# Copyright 2023 Anirudh Singh
# SPDX-License-Identifier: BSD-3-Clause

"""Provide persistence, feature, sequence, and distribution helpers."""

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

from forecasting_optimizer import constants


def identify_isolation_forest_outliers(data):
    """Split a DataFrame by an isolation forest's outlier predictions.

    Args:
        data: pandas DataFrame containing the observations to classify.

    Returns:
        A tuple ``(normal_rows, outlier_rows, outlier_mask)``. The first two
        values are DataFrames and the final value is a Boolean NumPy array.
    """
    clf = IsolationForest(n_estimators=100, warm_start=True)
    clf.fit(data)

    y_pred = clf.predict(data)
    y_pred = pd.Series(y_pred)
    if -1 in y_pred.values:
        # IsolationForest uses -1 for outliers and 1 for inliers.
        ol_count = y_pred.value_counts()[-1]
    else:
        ol_count = 0
    logging.warning(f"The total number of outliers are: {ol_count}")
    mask = (y_pred == -1).values
    data_ol = data.loc[mask, :]
    data_norm = data.loc[~mask, :]
    return data_norm, data_ol, mask


def nested_dict_to_flatten_df(dc, levels, value_col_name="values"):
    """Flatten a nested dictionary into a pandas DataFrame.

    Args:
        dc: Dictionary whose nesting follows the order in ``levels``.
        levels: Ordered column names for the dictionary's nesting levels.
        value_col_name: Name assigned to the leaf-value column.

    Returns:
        A DataFrame with one column per level and one value column.
    """
    top_level = levels[0]
    keys = list(dc.keys())
    temp_dfs = list()
    if len(levels) > 1:
        for key in keys:
            remaining_levels = levels[1:]
            temp_df = nested_dict_to_flatten_df(
                dc=dc[key],
                levels=remaining_levels,
                value_col_name=value_col_name,
            )
            temp_df[top_level] = key
            temp_dfs.append(temp_df)
        df = pd.concat(temp_dfs, axis=0)
    else:
        df = pd.Series(dc).to_frame().reset_index()
        df.columns = [top_level, value_col_name]
    df = df[levels + [value_col_name]]
    return df


def save_df(data, experiment_id, name):
    """Persist an experiment DataFrame in HDF format.

    Args:
        data: pandas DataFrame to persist.
        experiment_id: Identifier used to namespace the output file.
        name: Artifact name used in both the file name and HDF key.
    """
    file_name = f"{constants.OUTPUT_DATA_FOLDER}/{experiment_id}_{name}.hdf"
    data.to_hdf(file_name, key=name)


def read_df(experiment_id, name):
    """Load an experiment DataFrame from its HDF artifact.

    Args:
        experiment_id: Identifier used to namespace the output file.
        name: Artifact name used in both the file name and HDF key.

    Returns:
        The pandas DataFrame stored for the experiment artifact.
    """
    file_name = f"{constants.OUTPUT_DATA_FOLDER}/{experiment_id}_{name}.hdf"
    data = pd.read_hdf(file_name, key=name)
    return data


def save_data(data, experiment_id, name, name_len_limit=100):
    """Serialize an experiment artifact with pickle.

    Args:
        data: Python object to serialize.
        experiment_id: Identifier used to namespace the output file.
        name: Artifact name included in the output file name.
        name_len_limit: Maximum number of artifact-name characters to retain.
    """
    file_name = (
        f"{constants.OUTPUT_DATA_FOLDER}/"
        f"{experiment_id}_{name[:name_len_limit]}.pickle"
    )
    with open(file_name, "wb") as f:
        pickle.dump(data, f)


def read_data(experiment_id, name):
    """Deserialize a trusted experiment artifact from pickle.

    Args:
        experiment_id: Identifier used to namespace the output file.
        name: Artifact name included in the input file name.

    Returns:
        The Python object stored in the artifact.

    Warning:
        Pickle files can execute code while loading and must come from a
        trusted source.
    """
    file_name = f"{constants.OUTPUT_DATA_FOLDER}/{experiment_id}_{name}.pickle"
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    return data


def getBoxplotOuliters(s: pd.Series):  # pylint: disable=invalid-name
    """Calculate Tukey boxplot fences for a pandas Series.

    The public name, including its historical spelling, is retained for caller
    compatibility.

    Args:
        s: Numeric pandas Series.

    Returns:
        A tuple ``(lower_limit, upper_limit)`` containing the boxplot fences.
    """
    p25 = s.quantile(0.25)
    p75 = s.quantile(0.75)
    iqr = p75 - p25

    upper_limit = p75 + 1.5 * iqr
    lower_limit = p25 - 1.5 * iqr

    return lower_limit, upper_limit


def deepsum(obj, exclude_dict_keys=[], ignore_na=True):
    """Recursively sum numeric leaves in supported lists and dictionaries.

    The default exclusion list is treated as immutable.
    This legacy implementation accesses ``np.int`` and ``np.float``, which are
    unavailable in current NumPy releases and cause ``AttributeError``.

    Args:
        obj: Numeric scalar, list, tuple, or dictionary to traverse.
        exclude_dict_keys: Dictionary keys whose values should be skipped.
        ignore_na: Whether to omit missing numeric values from the sum.

    Returns:
        The recursive numeric total, or zero when no supported leaves exist.
    """
    if type(obj) in [
        int,
        float,
        np.int,
        np.int32,
        np.int64,
        np.float,
        np.float32,
        np.float64,
    ]:
        return obj

    sum = 0
    if type(obj) == dict:
        for key, value in obj.items():
            if key in exclude_dict_keys:
                continue
            val = deepsum(
                obj=value,
                exclude_dict_keys=exclude_dict_keys,
                ignore_na=ignore_na,
            )
            # Preserve the legacy missing-value predicate and evaluation order.
            if (val in [None, np.nan]) | np.isnan(val):
                if ignore_na:
                    continue
                else:
                    sum += val
            else:
                sum += val

    if type(obj) in [list, np.array, tuple]:
        for value in obj:
            val = deepsum(
                obj=value,
                exclude_dict_keys=exclude_dict_keys,
                ignore_na=ignore_na,
            )
            # Preserve the legacy missing-value predicate and evaluation order.
            if (val in [None, np.nan]) | np.isnan(val):
                if ignore_na:
                    continue
                else:
                    sum += val
            else:
                sum += val

    return sum


def deepsum_with_len(obj, exclude_dict_keys=[], ignore_na=True):
    """Recursively sum numeric leaves and count the values included.

    The default exclusion list is treated as immutable.
    This legacy implementation accesses ``np.int`` and ``np.float``, which are
    unavailable in current NumPy releases and cause ``AttributeError``.

    Args:
        obj: Numeric scalar, list, tuple, or dictionary to traverse.
        exclude_dict_keys: Dictionary keys whose values should be skipped.
        ignore_na: Whether to omit missing numeric values from the result.

    Returns:
        A tuple ``(total, value_count)`` for the supported numeric leaves.
    """
    if type(obj) in [
        int,
        float,
        np.int,
        np.int32,
        np.int64,
        np.float,
        np.float32,
        np.float64,
    ]:
        return obj, 1

    sum = 0
    len = 0
    if type(obj) == dict:
        for key, value in obj.items():
            if key in exclude_dict_keys:
                continue
            val, l = deepsum_with_len(
                obj=value,
                exclude_dict_keys=exclude_dict_keys,
                ignore_na=ignore_na,
            )
            # Preserve the legacy missing-value predicate and evaluation order.
            if (val in [None, np.nan]) | np.isnan(val):
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
            val, l = deepsum_with_len(
                obj=value,
                exclude_dict_keys=exclude_dict_keys,
                ignore_na=ignore_na,
            )
            # Preserve the legacy missing-value predicate and evaluation order.
            if (val in [None, np.nan]) | np.isnan(val):
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
    """Add calendar features derived from a DataFrame index or column.

    The retained implementation accesses the removed pandas ``week``
    attribute and raises ``AttributeError`` on current pandas releases.

    Args:
        trial: Optimization-trial value accepted for pipeline API consistency.
        df: pandas DataFrame to mutate with calendar-feature columns.
        date_col: Optional source column; when omitted, use the DataFrame index.

    Returns:
        The same DataFrame object with day, weekday, week, and month columns.
    """
    if date_col is None:
        dates = df.index
        dates = pd.to_datetime(dates)
    else:
        dates = df[date_col]
        dates = pd.to_datetime(dates).dt

    df["day"] = dates.day
    df["dayofweek"] = dates.dayofweek
    df["week"] = dates.week
    df["month"] = dates.month

    return df


def flatten(list_of_lists):
    """Recursively flatten nested Python lists.

    Args:
        list_of_lists: List whose nested list elements should be flattened.

    Returns:
        A flat list containing the input's non-list elements in traversal order.
    """
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def treat_categorical_features(
    encoder, train, val, test, categorical_features=None
):
    """Fit an encoder and replace categorical columns in dataset splits.

    This function mutates a supplied encoder by fitting it. It returns new
    DataFrames assembled from the retained and encoded columns.

    Args:
        encoder: One-hot-compatible encoder, or ``None`` to create one.
        train: Training pandas DataFrame used to fit the encoder.
        val: Optional validation DataFrame to transform.
        test: Optional test DataFrame to transform.
        categorical_features: Optional iterable of columns to encode. When
            omitted, all non-numeric training columns are selected.

    Returns:
        A tuple ``(encoder, encoded_columns, train, val, test)`` containing the
        fitted encoder, generated column names, and transformed splits.
    """
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=1)

    if categorical_features is None:
        categorical_features = train.select_dtypes(
            exclude=np.number
        ).columns.tolist()

    # A set removes duplicate feature names before the encoder is fitted.
    categorical_features = list(set(categorical_features))
    encoder.fit(train[categorical_features])
    encoded_columns = [
        x if y is None else [e for e in x if e not in y] + ["infrequent_val"]
        for x, y in tuple(
            zip(encoder.categories_, encoder.infrequent_categories_)
        )
    ]
    encoded_columns = dict(zip(categorical_features, encoded_columns))
    encoded_columns = [
        [f"{col}_{val}" for val in list(encoded_values)]
        for col, encoded_values in encoded_columns.items()
    ]
    encoded_columns = np.concatenate(encoded_columns)

    # Preserve split indexes so encoded rows align during concatenation.
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

    # Concatenate encoded features with the original non-categorical columns.
    train = pd.concat(
        [train.drop(columns=categorical_features), train_encoded], axis=1
    )
    if val is not None:
        val = pd.concat(
            [val.drop(columns=categorical_features), val_encoded], axis=1
        )
    if test is not None:
        test = pd.concat(
            [test.drop(columns=categorical_features), test_encoded], axis=1
        )

    return encoder, encoded_columns, train, val, test


def get_week_from_date(df, date_col):
    """Add Sunday-based week-start dates to a DataFrame.

    Args:
        df: pandas DataFrame to mutate.
        date_col: Name of the source date column, converted to datetime.

    Returns:
        The same DataFrame object with a ``week`` column of date objects.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    var = df[date_col].dt.dayofweek + 1
    var = var.mask(var == 7, 0)
    df["week"] = df[date_col] - pd.to_timedelta(var, unit="days")
    df["week"] = df["week"].dt.date
    return df


def deepbrowse(obj, func, exclude_dict_keys=[]):
    """Call a function for every supported numeric leaf in a nested object.

    The default exclusion list is treated as immutable.
    This legacy implementation accesses ``np.int`` and ``np.float``, which are
    unavailable in current NumPy releases and cause ``AttributeError``.

    Args:
        obj: Numeric scalar, list, tuple, or dictionary to traverse.
        func: Callable invoked once for each supported numeric leaf.
        exclude_dict_keys: Dictionary keys whose values should be skipped.
    """
    if type(obj) in [
        int,
        float,
        np.int,
        np.int32,
        np.int64,
        np.float,
        np.float32,
        np.float64,
    ]:
        func(obj)

    if type(obj) == dict:
        for key, value in obj.items():
            if key in exclude_dict_keys:
                continue
            deepbrowse(
                obj=value, func=func, exclude_dict_keys=exclude_dict_keys
            )

    if type(obj) in [list, np.array, tuple]:
        for value in obj:
            deepbrowse(
                obj=value, func=func, exclude_dict_keys=exclude_dict_keys
            )


def kl_divergence(actual_probabilities, predicted_probabilities, base=2):
    """Calculate the custom Kullback-Leibler divergence used by the framework.

    Args:
        actual_probabilities: Iterable containing the reference probabilities.
        predicted_probabilities: Iterable containing comparison probabilities.
        base: Logarithm base used in the divergence calculation.

    Returns:
        The negated accumulated divergence value.
    """
    e = 0
    for pp, ap in zip(predicted_probabilities, actual_probabilities):
        e = e + ap * (math.log(pp / ap, base))

    return -e


def js_divergence(actual_density, pred_density, version="auf", base=2):
    """Calculate Jensen-Shannon divergence with the selected implementation.

    This returns divergence rather than the square-root Jensen-Shannon distance.
    The TensorFlow branch expects a module-level ``tf`` binding, which this
    module does not create; callers must inject it for that branch to run.

    Args:
        actual_density: Iterable containing the reference probability density.
        pred_density: Iterable containing the comparison probability density.
        version: ``'auf'`` for the local KL implementation or ``'tf'`` for a
            TensorFlow ``KLDivergence`` implementation.
        base: Logarithm base used by the local implementation.

    Returns:
        The Jensen-Shannon divergence between the two densities.

    Raises:
        NameError: If ``version`` is ``'tf'`` and no ``tf`` binding was
            injected.
    """
    actual_density = np.array(actual_density)
    pred_density = np.array(pred_density)
    m = (actual_density + pred_density) / 2
    if version == "auf":
        p = kl_divergence(actual_density, m, base=base)
        q = kl_divergence(pred_density, m, base=base)
    elif version == "tf":
        kl = tf.keras.losses.KLDivergence()
        p = kl(list(actual_density), list(m)).numpy()
        q = kl(list(pred_density), list(m)).numpy()

    jsd = (p + q) / 2
    return jsd


def get_templated_distribution(df, col, prob_template):
    """Align a column's empirical distribution with a probability template.

    Args:
        df: pandas DataFrame containing the observed values.
        col: Column whose normalized value counts should be calculated.
        prob_template: pandas Series whose index defines the output classes.

    Returns:
        A pandas Series aligned to the template's index.
    """
    vc = df[col].value_counts(normalize=True)
    vc = vc + prob_template
    vc = vc.fillna(0)[prob_template.index]
    return vc


def get_jsd(s, classes):
    """Calculate row-level Jensen-Shannon divergence for class probabilities.

    This helper selects the TensorFlow branch of ``js_divergence``. It therefore
    requires callers to inject a module-level ``tf`` binding before use.

    Args:
        s: pandas Series containing an actual distribution and class scores.
        classes: Ordered class labels used to select predicted probabilities.

    Returns:
        The TensorFlow-backed Jensen-Shannon divergence for the row.

    Raises:
        NameError: If no module-level ``tf`` binding was injected.
    """
    vc = pd.Series(s[constants.ACTUAL_DISTRIBUTION_STR])
    prob_template = pd.Series(data=0, index=classes)
    vc = prob_template + vc
    vc = vc.fillna(0)
    probs = pd.concat([vc, s[classes]], axis=1).fillna(0)
    jsd = js_divergence(
        actual_density=probs.iloc[:, 0],
        pred_density=probs.iloc[:, 1],
        version="tf",
        base=2,
    )
    return jsd


def get_nb_dist(mean, std):
    """Construct a SciPy negative-binomial distribution from moments.

    Args:
        mean: Desired distribution mean.
        std: Desired distribution standard deviation.

    Returns:
        A frozen SciPy negative-binomial distribution.
    """
    p = mean / (std**2)
    n = (mean**2) / (std**2 - mean)
    dist = nbinom(n, p)
    return dist


def get_nb_probs(dist, mean, std, num_std_spread):
    """Sample and normalize negative-binomial probabilities around a mean.

    Args:
        dist: Distribution object exposing a ``pmf`` method.
        mean: Center of the sampled integer support.
        std: Standard deviation used to size the support.
        num_std_spread: Number of standard deviations sampled on each side.

    Returns:
        A pandas Series of normalized probabilities indexed by support value.
    """
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
    """Sample and normalize continuous density values around a mean.

    Args:
        dist: Distribution object exposing a ``pdf`` method.
        mean: Center of the sampled integer support.
        std: Standard deviation used to size the support.
        num_std_spread: Number of standard deviations sampled on each side.

    Returns:
        A pandas Series of normalized density values indexed by support value.
    """
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
    """Convert dates to pandas weekly-period start timestamps.

    Depending on the input and ``index`` flag, this function mutates an object
    index, replaces a Series, or mutates a DataFrame column.

    Args:
        s: pandas Series or DataFrame containing dates.
        index: Whether to convert the object's index instead of contained
            values.
        date_col: DataFrame column to convert when ``index`` is false.

    Returns:
        The converted pandas object, or the original object for unsupported
        input types.
    """
    if index:
        days = pd.to_datetime(pd.Series(s.index))
        s.index = days.dt.to_period("W").apply(lambda r: r.start_time)
    else:
        if isinstance(s, pd.Series):
            days = pd.to_datetime(s)
            s = days.dt.to_period("W").apply(lambda r: r.start_time)
        elif isinstance(s, pd.DataFrame):
            days = pd.to_datetime(s[date_col])
            s[date_col] = days.dt.to_period("W").apply(lambda r: r.start_time)
        else:
            logging.error(f"argument s is not a valid object type.")
    return s


def cs_to_seq(x, y, length, sampling_rate, stride):
    """Transform cross-sectional arrays into time-series sequences.

    Args:
        x: Input samples accepted by Keras ``TimeseriesGenerator``.
        y: Targets aligned with ``x``.
        length: Number of historical timesteps per generated sample.
        sampling_rate: Interval between timesteps within a sequence.
        stride: Interval between consecutive generated sequences.

    Returns:
        A tuple ``(sequences, targets)`` containing NumPy arrays.
    """
    data_gen = TimeseriesGenerator(
        x,
        y,
        length=length,
        sampling_rate=sampling_rate,
        batch_size=1,
        stride=stride,
    )
    x = np.array([seq for seq, _ in data_gen]).squeeze()
    y = np.array([target for _, target in data_gen])
    return x, y


def get_lags(trial, df, cols, lags=[1], dropna=True):
    """Add lagged columns and optionally remove their leading incomplete rows.

    The default lag list is treated as immutable. The input DataFrame is
    mutated before an optional sliced view or copy is returned.

    Args:
        trial: Optimization-trial value accepted for pipeline API consistency.
        df: pandas DataFrame to receive lagged columns.
        cols: Iterable of source column names.
        lags: Shared lag iterable, or mapping from columns to lag iterables.
        dropna: Whether to remove the first ``max(lags)`` rows.

    Returns:
        The DataFrame containing the generated lag columns.
    """
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

    return df


def get_diffs(trial, df, cols, diffs=[1], dropna=True):
    """Add differenced columns and optionally remove initial incomplete rows.

    The default difference list is treated as immutable. The input DataFrame is
    mutated before an optional sliced view or copy is returned.

    Args:
        trial: Optimization-trial value accepted for pipeline API consistency.
        df: pandas DataFrame to receive differenced columns.
        cols: Iterable of source column names.
        diffs: Shared period iterable, or mapping from columns to period
            iterables.
        dropna: Whether to remove the first ``max(diffs)`` rows.

    Returns:
        The DataFrame containing the generated difference columns.
    """
    for col in cols:
        if isinstance(diffs, dict):
            col_diffs = diffs[col]
        else:
            col_diffs = diffs

        for diff in col_diffs:
            df[f"{col}_diff{diff}"] = df[col].diff(periods=diff)

    if dropna:
        if isinstance(diffs, dict):
            max_diff = np.concatenate(list(diffs.values())).max()
        else:
            max_diff = np.array(diffs).max()

        df = df.iloc[max_diff:, :]

    return df
