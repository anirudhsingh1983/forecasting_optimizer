# Copyright 2023 Anirudh Singh
# SPDX-License-Identifier: BSD-3-Clause

"""Load and aggregate logistics data for end-to-end forecasting."""

import logging
import pickle
import warnings
from importlib import resources

import pandas as pd

from forecasting_optimizer.projects.e2e import constants as const
from forecasting_optimizer.projects.e2e import settings
from forecasting_optimizer.util import gbq

# Preserve the process-wide warning and display behavior expected by existing
# exploratory workflows that import this module.
warnings.filterwarnings("ignore")
pd.set_option("display.expand_frame_repr", True)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_info_columns", 500)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)


def get_data(
    isc_only=True,
    local=settings.READ_DATA_LOCAL,
    leg=const.FC_ID_COL,
    move=const.ARRIVAL_STR,
):
    """Load shipments and aggregate a selected leg column by event date.

    Args:
        isc_only: Whether to retain only ISC service levels.
        local: Whether to load cached data instead of querying BigQuery.
        leg: Column whose values are summed for each event date.
        move: Event type used to select the leg's arrival or departure date.

    Returns:
        A date-indexed DataFrame whose ``units`` column contains the selected
        leg column's aggregated values.
    """
    if local:
        with open(f"{const.LOCAL_DATA_DIR}/base_data.pickle", "rb") as f:
            data = pickle.load(f)
    else:
        query = (
            resources.files(__package__)
            .joinpath("queries", "spo_data_3.sql")
            .read_text(encoding="utf-8")
        )

        data = gbq.run_gbq_query(
            query,
            project="wf-gcp-us-ae-dsservice-prod",
            progress_bar_type="tqdm",
            use_cache=True,
        )
        with open(f"{const.LOCAL_DATA_DIR}/base_data.pickle", "wb") as f:
            pickle.dump(data, f)

    name_mapping = {
        "SPOID": const.SPOID_COL,
        "servicelevel": const.SERVICELEVEL_COL,
        "Original_WH": const.FC_ID_COL,
        "Original_WH_Name": const.FC_NAME_COL,
        "OriginPortCode": const.ORIGIN_PORT_NAME_COL,
        "DestinationPortCode": const.DESTINATION_PORT_NAME_COL,
        "arrivaldate": const.FC_ARRIVAL_TIMESTAMP,
        "Total_units": const.QUANTITY_COL,
    }
    for col in data.columns:
        perc = 100 * data[col].isna().sum() / len(data)
        if perc > settings.MISSING_VAL_WARNING_PERCENT:
            logging.warning(
                f"Percentage of missing values in {col}: {perc.round(2)}%"
            )

    data = data.rename(columns=name_mapping)

    data = data[data[const.QUANTITY_COL] > 0]

    date_cols = [
        const.ACF_ARRIVAL_TIMESTAMP,
        const.ACF_DEPARTURE_TIMESTAMP,
        const.ORIGIN_PORT_ARRIVAL_TIMESTAMP,
        const.ORIGIN_PORT_DEPARTURE_TIMESTAMP,
        const.DESTINATION_PORT_ARRIVAL_TIMESTAMP,
        const.DESTINATION_PORT_DEPARTURE_TIMESTAMP,
        const.FC_ARRIVAL_TIMESTAMP,
    ]

    data[date_cols] = data[date_cols].apply(
        lambda x: pd.to_datetime(x).dt.date, axis=0
    )

    for col in [
        const.ACF_LOCATION_NAME_COL,
        const.ORIGIN_PORT_NAME_COL,
        const.DESTINATION_PORT_NAME_COL,
        const.FC_NAME_COL,
    ]:
        data[col] = data[col].str.strip()

    data[const.FC_ID_COL] = data[const.FC_ID_COL].astype(str)

    if isc_only:
        isc_mask = data[const.SERVICELEVEL_STR].isin(const.isc_sls)
        df = data[isc_mask]
    else:
        df = data

    # Derive transit and dwell durations before converting timedeltas to days.
    df[const.STAY_AT_CSF_STR] = (
        df[const.ACF_DEPARTURE_TIMESTAMP] - df[const.ACF_ARRIVAL_TIMESTAMP]
    )
    df[const.CSF_TO_ORIGIN_PORT_STR] = (
        df[const.ORIGIN_PORT_ARRIVAL_TIMESTAMP]
        - df[const.ACF_DEPARTURE_TIMESTAMP]
    )
    df[const.STAY_AT_ORIGIN_PORT_STR] = (
        df[const.ORIGIN_PORT_DEPARTURE_TIMESTAMP]
        - df[const.ORIGIN_PORT_ARRIVAL_TIMESTAMP]
    )
    df[const.ORIGIN_PORT_TO_DESTINATION_PORT_STR] = (
        df[const.DESTINATION_PORT_ARRIVAL_TIMESTAMP]
        - df[const.ORIGIN_PORT_DEPARTURE_TIMESTAMP]
    )
    df[const.STAY_AT_DESTINATION_PORT_STR] = (
        df[const.DESTINATION_PORT_DEPARTURE_TIMESTAMP]
        - df[const.DESTINATION_PORT_ARRIVAL_TIMESTAMP]
    )
    df[const.DESTINATION_PORT_TO_FC_STR] = (
        df[const.FC_ARRIVAL_TIMESTAMP]
        - df[const.DESTINATION_PORT_DEPARTURE_TIMESTAMP]
    )

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
        port_mappings = (
            port_mappings.drop(columns=[const.ORIGIN_PORT_NAME_COL])
            .squeeze()
            .to_dict()
        )
        df[const.ORIGIN_PORT_NAME_COL] = df[const.ORIGIN_PORT_NAME_COL].map(
            port_mappings
        )

    node_date_mapping = {
        const.ARRIVAL_STR: {
            const.ACF_LOCATION_NAME_COL: const.ACF_ARRIVAL_TIMESTAMP,
            const.ORIGIN_PORT_NAME_COL: const.ORIGIN_PORT_ARRIVAL_TIMESTAMP,
            const.DESTINATION_PORT_NAME_COL: (
                const.DESTINATION_PORT_ARRIVAL_TIMESTAMP
            ),
            const.FC_KEY: const.FC_ARRIVAL_TIMESTAMP,
        },
        const.DEPARTURE_STR: {
            const.ACF_LOCATION_NAME_COL: const.ACF_DEPARTURE_TIMESTAMP,
            const.ORIGIN_PORT_NAME_COL: const.ORIGIN_PORT_DEPARTURE_TIMESTAMP,
            const.DESTINATION_PORT_NAME_COL: (
                const.DESTINATION_PORT_DEPARTURE_TIMESTAMP
            ),
        },
    }

    date_col = node_date_mapping[move][leg]
    df = pd.Series(data=df[leg].astype(int).values, index=df[date_col])
    df = df.groupby(df.index).sum().to_frame().rename(columns={0: "units"})
    df = df.sort_index()
    return df
