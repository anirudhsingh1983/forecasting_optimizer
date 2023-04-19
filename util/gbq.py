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
    def __init__(self):
        self._cache = {}

    def contains(self, query: str) -> bool:
        key = self._get_key(query)
        return key in self._cache

    def get(self, query: str) -> pd.DataFrame:
        key = self._get_key(query)
        data = self._cache[key]
        return data.copy()  # storing dataframes, user might change them -> MUST always return a copy

    def set(self, query: str, data: pd.DataFrame) -> None:
        mutex.acquire()
        key = self._get_key(query)
        if key not in self._cache:
            self._cache[key] = data.copy()
        mutex.release()

    @staticmethod
    def _get_key(query: str) -> str:
        # todo when adding parametrized queries, add params to key!
        return hashlib.sha256(query.encode("UTF-8")).hexdigest()


def run_gbq_query(query: str,
                  project: str,
                  parse_dates: List[str] = None,
                  progress_bar_type: str = 'tqdm',
                  use_cache: bool = False):
    """
    Executes a GBQ query and returns the output of the query as a pandas dataframe.

    Example:
        from util import gbq
        query = "SELECT * FROM `wf-gcp-us-ae-ops-prod.csn_reporting_isc.datamart_combined` LIMIT 100"
        project='wf-gcp-us-ae-dsservice-prod'
        d = gbq.run_gbq_query(query=query,
                          project=project,
                          parse_dates = None,
                          progress_bar_type = 'tqdm',
                          use_cache = False)
    """

    _bqclient = bigquery.Client(project=project)
    _bqstorageclient = bigquery_storage.BigQueryReadClient()

    # cache of query results
    # always risky to have a cache, but on the other hand cumbersome to keep querying the same data
    # this is an attempt at finding a compromise between risk and convenience
    # the cache is not stored on file, only exists in memory
    # this means that for:
    #    * a script will only be valid during the script run, when re-running the script, also the queries will be rerun
    #    * a notebook it will be only valid per kernel, i.e. each notebook has a different cache, when restarting kernel
    #      or notebook server, the queries will be run again
    _cache = QueryCache()

    # already have a cached result for this query
    if use_cache and _cache.contains(query):
        log.debug("Reading data from cache")
        return _cache.get(query)

    # no cached result -> actually run the query
    log.info("Starting query")

    # run the query
    result_stream = _bqclient.query(query).result()
    log.info("Query done, now downloading the data")

    # download the data
    # use the_bqstorageclient to massively speed up data transfer!
    # see https://stackoverflow.com/questions/53432996/takes-too-long-to-export-data-from-bigquery-into-jupyter-notebook
    result = result_stream.to_dataframe(bqstorage_client=_bqstorageclient, progress_bar_type=progress_bar_type)

    # parse everything that should be dates
    if parse_dates:
        for col in parse_dates:
            result[col] = pd.to_datetime(result[col])

    # update the cache
    if use_cache:
        _cache.set(query, result)

    return result
