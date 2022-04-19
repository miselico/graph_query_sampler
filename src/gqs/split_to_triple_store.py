"""Functions to upload the dataseplits to triple stores"""

import json
import pathlib
import time
import requests
import uuid
import logging

from .dataset import Dataset

__all__ = [
    "create_graphdb_repository", "remove_graphdb_repository", "store_triples"
]

logger = logging.getLogger(__name__)


def create_graphdb_repository(repositoryID: str, graphdb_url: str):
    url = f"{graphdb_url}/rest/repositories"
    files = {'config': ('config.ttl', f'''
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>.
@prefix rep: <http://www.openrdf.org/config/repository#>.
@prefix sr: <http://www.openrdf.org/config/repository/sail#>.
@prefix sail: <http://www.openrdf.org/config/sail#>.
@prefix owlim: <http://www.ontotext.com/trree/owlim#>.

[] a rep:Repository ;
    rep:repositoryID "{repositoryID}" ;
    rdfs:label "" ;
    rep:repositoryImpl [
        rep:repositoryType "graphdb:FreeSailRepository" ;
        sr:sailImpl [
            sail:sailType "graphdb:FreeSail" ;

            owlim:base-URL "http://example.org/owlim#" ;
            owlim:defaultNS "" ;
            owlim:entity-index-size "10000000" ;
            owlim:entity-id-size  "32" ;
            owlim:imports "" ;
            owlim:repository-type "file-repository" ;
            owlim:ruleset "empty" ;
            owlim:storage-folder "storage" ;

            owlim:enable-context-index "false" ;

            owlim:enablePredicateList "true" ;

            owlim:in-memory-literal-properties "true" ;
            owlim:enable-literal-index "true" ;

            owlim:check-for-inconsistencies "false" ;
            owlim:disable-sameAs  "true" ;
            owlim:query-timeout  "0" ;
            owlim:query-limit-results  "0" ;
            owlim:throw-QueryEvaluationException-on-timeout "false" ;
            owlim:read-only "false" ;
        ]
    ].
    ''')}
    response = requests.post(url=url, files=files)
    if response.status_code != 201:
        raise Exception(f"Creating the repository failed. does it alreade exist? Message: {str(response.content)}")


def store_triples(dataset: Dataset, data: pathlib.Path, graphname: str, graphdb_url: str):
    repositoryID = dataset.graphDB_repositoryID()
    url = f"{graphdb_url}/rest/data/import/upload/{repositoryID}/file"

    unique_name = str(uuid.uuid4()) + "-data.nt"

    import_settings = {"name": unique_name, "status": "NONE", "message": "", "context": graphname,
                       "replaceGraphs": [], "baseURI": None, "forceSerial": False, "type": None,
                       "format": None, "data": "somedatatotest", "timestamp": 0,
                       "parserSettings": {
                           "preserveBNodeIds": False, "failOnUnknownDataTypes": False, "verifyDataTypeValues": False,
                           "normalizeDataTypeValues": False, "failOnUnknownLanguageTags": False, "verifyLanguageTags": True,
                           "normalizeLanguageTags": False, "stopOnError": True
                       },
                       "requestIdHeadersToForward": None,
                       }
    import_setting_json = json.dumps(import_settings)
    files = {
        'importSettings': ('blob', import_setting_json, 'application/json'),
        'file': ('data.nt', open(data, 'rb'), 'application/octet-stream')
    }
    response = requests.post(url=url, files=files)
    if response.status_code != 202:
        raise Exception(f"Unexpected response from triple store. Uploading the file failed: {str(response.content)}")
    logger.info("Started importing {data} into repository: {repositoryID} graph: {graphname}. Waiting for import to finish")
    while True:
        time.sleep(1.0)
        url = f"{graphdb_url}/rest/data/import/upload/{repositoryID}"
        resp = requests.get(url)
        import_infos = json.loads(resp.content)
        for import_info in import_infos:
            file_name = import_info["name"]
            if not file_name == unique_name:
                continue
            # We are looking at the record for the right file now
            status = import_info["status"]
            if status == "DONE":
                message = import_info["message"]
                logger.info(f"Finished upload: {message} Removing temp file from triple store.")
                url = f"{graphdb_url}/rest/data/import/upload/{repositoryID}/status?remove=true"
                payload = json.dumps([unique_name])
                resp = requests.delete(url=url, headers={"Content-type": "application/json;charset=UTF-8"}, data=payload)
                if resp.status_code != 200:
                    raise Exception("Removing the file failed. Maybe something went wrong.")
                logger.info(f"Temp file removed. Done importing: {data} into repository: {repositoryID} graph: {graphname} ")
                return
            elif status == "ERROR":
                message = import_info["message"]
                raise Exception(f"An error happened uploading the file {data} : {message}")
            elif status == "IMPORTING":
                logger.info("Still importing")
            else:
                raise Exception("Unknown upoad status")


def remove_graphdb_repository(dataset: Dataset, graphdb_url: str):
    repositoryID = dataset.graphDB_repositoryID()
    url = f"{graphdb_url}/rest/repositories/{repositoryID}"
    print(url)
    response = requests.delete(url)
    if response.status_code != 200:
        raise Exception(str(response.content))
