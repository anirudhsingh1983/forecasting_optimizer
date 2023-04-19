import logging
import pickle
import warnings

import pandas as pd

from projects.e2e import constants as const
from projects.e2e import settings
from util import gbq

warnings.filterwarnings("ignore")
pd.set_option('display.expand_frame_repr', True)
pd.set_option('display.max_columns', None)  # Set to None for unlimited number of output rows.
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_info_columns', 500)
pd.set_option('display.max_rows', None)  # Set to None for unlimited number of output rows.
pd.set_option('display.width', 500)  # Width of the display in characters.


def get_data(isc_only=True, local=settings.READ_DATA_LOCAL, leg = const.FC_ID_COL, move=const.ARRIVAL_STR):
    if local:
        with open(f'{const.LOCAL_DATA_DIR}/base_data.pickle', 'rb') as f:
            data = pickle.load(f)
    else:
        with open('projects/e2e/queries/spo_data_3.sql', 'r') as f:
            query = f.read()

        data = gbq.run_gbq_query(query,
                                 project="wf-gcp-us-ae-dsservice-prod",
                                 progress_bar_type='tqdm',
                                 use_cache=True)
        with open(f'{const.LOCAL_DATA_DIR}/base_data.pickle', 'wb') as f:
            pickle.dump(data, f)

    name_mapping = {
        'SPOID': const.SPOID_COL,
        'servicelevel': const.SERVICELEVEL_COL,
        'Original_WH': const.FC_ID_COL,
        'Original_WH_Name': const.FC_NAME_COL,
        'OriginPortCode': const.ORIGIN_PORT_NAME_COL,
        'DestinationPortCode': const.DESTINATION_PORT_NAME_COL,
        'arrivaldate': const.FC_ARRIVAL_TIMESTAMP,
        'Total_units': const.QUANTITY_COL,
    }
    for col in data.columns:
        perc = 100 * data[col].isna().sum() / len(data)
        if perc > settings.MISSING_VAL_WARNING_PERCENT:
            logging.warning(f"Percentage of missing values in {col}: {perc.round(2)}%")

    data = data.rename(columns=name_mapping)

    data = data[data[const.QUANTITY_COL] > 0]

    date_cols = [
        const.ACF_ARRIVAL_TIMESTAMP, const.ACF_DEPARTURE_TIMESTAMP, const.ORIGIN_PORT_ARRIVAL_TIMESTAMP,
        const.ORIGIN_PORT_DEPARTURE_TIMESTAMP, const.DESTINATION_PORT_ARRIVAL_TIMESTAMP,
        const.DESTINATION_PORT_DEPARTURE_TIMESTAMP, const.FC_ARRIVAL_TIMESTAMP
    ]

    data[date_cols] = data[date_cols].apply(lambda x: pd.to_datetime(x).dt.date, axis=0)

    for col in [const.ACF_LOCATION_NAME_COL, const.ORIGIN_PORT_NAME_COL, const.DESTINATION_PORT_NAME_COL,
                const.FC_NAME_COL]:
        data[col] = data[col].str.strip()

    data[const.FC_ID_COL] = data[const.FC_ID_COL].astype(str)

    if isc_only:
        isc_mask = data[const.SERVICELEVEL_STR].isin(const.isc_sls)
        df = data[isc_mask]
    else:
        df = data

    # transit and dwell times
    df[const.STAY_AT_CSF_STR] = df[const.ACF_DEPARTURE_TIMESTAMP] - df[const.ACF_ARRIVAL_TIMESTAMP]
    df[const.CSF_TO_ORIGIN_PORT_STR] = df[const.ORIGIN_PORT_ARRIVAL_TIMESTAMP] - df[const.ACF_DEPARTURE_TIMESTAMP]
    df[const.STAY_AT_ORIGIN_PORT_STR] = df[const.ORIGIN_PORT_DEPARTURE_TIMESTAMP] - df[
        const.ORIGIN_PORT_ARRIVAL_TIMESTAMP]
    df[const.ORIGIN_PORT_TO_DESTINATION_PORT_STR] = df[const.DESTINATION_PORT_ARRIVAL_TIMESTAMP] - df[
        const.ORIGIN_PORT_DEPARTURE_TIMESTAMP]
    df[const.STAY_AT_DESTINATION_PORT_STR] = df[const.DESTINATION_PORT_DEPARTURE_TIMESTAMP] - df[
        const.DESTINATION_PORT_ARRIVAL_TIMESTAMP]
    df[const.DESTINATION_PORT_TO_FC_STR] = df[const.FC_ARRIVAL_TIMESTAMP] - df[
        const.DESTINATION_PORT_DEPARTURE_TIMESTAMP]

    journey_cols = [
        const.CSF_TO_ORIGIN_PORT_STR,
        const.ORIGIN_PORT_TO_DESTINATION_PORT_STR,
        const.DESTINATION_PORT_TO_FC_STR,
        const.STAY_AT_CSF_STR,
        const.STAY_AT_ORIGIN_PORT_STR,
        const.STAY_AT_DESTINATION_PORT_STR,
    ]
    for col in journey_cols:
        df[col] = df[col].dt.days

    if settings.CONSOLIDATE_ORIGIN_PORTS:
        port_mappings = pd.read_csv(f"data/origin_port_regions.csv")
        port_mappings = port_mappings.drop_duplicates()
        port_mappings.index = port_mappings[const.ORIGIN_PORT_NAME_COL]
        port_mappings = port_mappings.drop(columns = [const.ORIGIN_PORT_NAME_COL]).squeeze().to_dict()
        df[const.ORIGIN_PORT_NAME_COL] = df[const.ORIGIN_PORT_NAME_COL].map(port_mappings)

    node_date_mapping = {
        const.ARRIVAL_STR: {
            const.ACF_LOCATION_NAME_COL: const.ACF_ARRIVAL_TIMESTAMP,
            const.ORIGIN_PORT_NAME_COL: const.ORIGIN_PORT_ARRIVAL_TIMESTAMP,
            const.DESTINATION_PORT_NAME_COL: const.DESTINATION_PORT_ARRIVAL_TIMESTAMP,
            const.FC_KEY: const.FC_ARRIVAL_TIMESTAMP
        },
        const.DEPARTURE_STR: {
            const.ACF_LOCATION_NAME_COL: const.ACF_DEPARTURE_TIMESTAMP,
            const.ORIGIN_PORT_NAME_COL: const.ORIGIN_PORT_DEPARTURE_TIMESTAMP,
            const.DESTINATION_PORT_NAME_COL: const.DESTINATION_PORT_DEPARTURE_TIMESTAMP,
        }
    }

    date_col = node_date_mapping[move][leg]
    df = pd.Series(data = df[leg].astype(int).values, index = df[date_col])
    df = df.groupby(df.index).sum().to_frame().rename(columns={0: 'units'})
    df = df.sort_index()
    return df
