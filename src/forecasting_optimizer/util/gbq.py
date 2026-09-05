# Copyright 2023 Anirudh Singh
# SPDX-License-Identifier: BSD-3-Clause

"""Execute Google BigQuery queries and manage defensive result copies."""

import hashlib
import logging
import threading
from typing import List

import pandas as pd
from google.cloud import bigquery
from google.cloud import bigquery_storage

log = logging.getLogger(__name__)

mutex = threading.Lock()


class QueryCache:
    """An in-memory cache of BigQuery results keyed by query text."""

    def __init__(self):
        """Initialize an empty query-result cache."""
        self._cache = {}

    def contains(self, query: str) -> bool:
        """Return whether a result is cached for ``query``.

        Args:
            query: SQL query text.

        Returns:
            Whether the cache contains an entry for the query.
        """
        key = self._get_key(query)
        return key in self._cache

    def get(self, query: str) -> pd.DataFrame:
        """Return a defensive copy of the result cached for ``query``.

        Args:
            query: SQL query text whose result has already been cached.

        Returns:
            A copy of the cached pandas DataFrame.
        """
        key = self._get_key(query)
        data = self._cache[key]
        # Callers may mutate DataFrames, so do not expose the cached object.
        return data.copy()

    def set(self, query: str, data: pd.DataFrame) -> None:
        """Cache a defensive copy of ``data`` if ``query`` is not present.

        Args:
            query: SQL query text used to derive the cache key.
            data: Query result to cache.
        """
        mutex.acquire()
        key = self._get_key(query)
        if key not in self._cache:
            self._cache[key] = data.copy()
        mutex.release()

    @staticmethod
    def _get_key(query: str) -> str:
        """Create a stable cache key from SQL query text.

        Args:
            query: SQL query text to hash.

        Returns:
            The hexadecimal SHA-256 digest of ``query``.
        """
        # Parameter values must become part of this key if parameters are added.
        return hashlib.sha256(query.encode("UTF-8")).hexdigest()


def run_gbq_query(
    query: str,
    project: str,
    parse_dates: List[str] = None,
    progress_bar_type: str = "tqdm",
    use_cache: bool = False,
):
    """Execute a BigQuery query and download its result into a DataFrame.

    Args:
        query: Google Standard SQL query text.
        project: Google Cloud project used to create the BigQuery client.
        parse_dates: Column names to convert to pandas datetime values.
        progress_bar_type: Progress-bar style passed to BigQuery's DataFrame
            conversion.
        use_cache: Whether to populate the newly created per-call cache. The
            cache is not reused by subsequent invocations.

    Returns:
        A pandas DataFrame containing the query result.
    """

    _bqclient = bigquery.Client(project=project)
    _bqstorageclient = bigquery_storage.BigQueryReadClient()

    # A fresh cache per invocation cannot reuse results from earlier calls.
    _cache = QueryCache()

    if use_cache and _cache.contains(query):
        log.debug("Reading data from cache")
        return _cache.get(query)

    log.info("Starting query")
    result_stream = _bqclient.query(query).result()
    log.info("Query done, now downloading the data")

    # The BigQuery Storage client substantially improves transfer throughput.
    result = result_stream.to_dataframe(
        bqstorage_client=_bqstorageclient,
        progress_bar_type=progress_bar_type,
    )

    if parse_dates:
        for col in parse_dates:
            result[col] = pd.to_datetime(result[col])

    if use_cache:
        _cache.set(query, result)

    return result
