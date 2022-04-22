"""internal functions for querying sparql triple stores, without producing output in the logger, except for errors"""

import logging
from typing import Any, Dict, cast

from rdflib.plugins.stores.sparqlstore import SPARQLStore
from rdflib.query import Result


def execute_csv_sparql_silenced(query: str, sparql_endpoint: str, sparql_endpoint_options: Dict[str, Any]) -> Result:
    store = SPARQLStore(sparql_endpoint, returnFormat="csv", method="POST", **sparql_endpoint_options)  # headers={}
    global_logger = logging.getLogger()
    original_level = global_logger.getEffectiveLevel()
    global_logger.setLevel(logging.ERROR)
    try:
        result = store.query(query)  # type: ignore
        return cast(Result, result)
    finally:
        global_logger.setLevel(original_level)


def execute_sparql_to_result_silenced(query: str, sparql_endpoint: str, sparql_endpoint_options: Dict[str, Any]) -> Result:
    store = SPARQLStore(sparql_endpoint, method="POST", **sparql_endpoint_options)  # headers={}
    global_logger = logging.getLogger()
    original_level = global_logger.getEffectiveLevel()
    # global_logger.setLevel(logging.ERROR)
    try:
        result = store.query(query)  # type: ignore
        return cast(Result, result)
    finally:
        global_logger.setLevel(original_level)
