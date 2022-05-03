"""Perform SPARQL queries to generate the dataset."""
import csv
import gzip
import hashlib
import json
import logging
import random
import re
import shutil
import traceback
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import (Any, DefaultDict, Dict, Generator, Iterable, List,
                    Optional, Tuple)

import pandas as pd
from gqs.dataset import Dataset
from rdflib import Variable
from requests import HTTPError

from ._sparql_execution import (execute_csv_sparql_silenced,
                                execute_sparql_to_result_silenced)

# from .config import formula_root, query_root, sparql_endpoint_address as default_sparql_endpoint_address, sparql_endpoint_options as default_sparql_endpoint_options

__all__ = [
    "preprocess_formulas",
    "sample_queries",
    "remove_queries"
]

logger = logging.getLogger(__name__)


def pairwise_directories(source_root: Path, destination_root: Path) -> Generator[Tuple[Path, Path], None, None]:
    """An iterator to iterate over recursive pairs of source and target directories,
    based on directories existing in the source directory and its not neceserily existing counterpart in the destination."""
    yield (source_root, destination_root)
    for potential_directory in source_root.iterdir():
        if potential_directory.is_dir():
            relative_directory = potential_directory.relative_to(source_root)
            destination_path = destination_root.joinpath(relative_directory)
            yield (potential_directory, destination_path)
            for (child_source, child_dest) in pairwise_directories(potential_directory, destination_path):
                yield (child_source, child_dest)


def preprocess_formulas(dataset: Dataset, sparql_endpoint: str, sparql_endpoint_options: Optional[Dict[str, Any]] = None) -> None:
    formula_source: Path = dataset.raw_formulas_location()
    formula_target: Path = dataset.formulas_location()
    sparql_endpoint_options = sparql_endpoint_options or {}

    for (source, target) in pairwise_directories(formula_source, formula_target):
        if not any(True for _ in source.glob("*.sparql")):
            # this directory has no sparql files, skip.
            continue
        # every folder with formulas must have its own config file
        configuration = _load_json_from_file(source / "config.json")
        # we keep a map from the variable names to the entities which are restricted
        excluded_entities: DefaultDict[str, set[str]] = defaultdict(set)
        for restriction in configuration["restrictions"]:
            if restriction["type"] == "max_indegree":
                query = '''
                    select ?entity  {
                    graph <split:all> {
                        ?s ?p ?entity .
                        }
                        FILTER(isIRI(?entity))
                    } group by ?entity
                    having (count(?entity) > ''' + str(restriction["maximum"]) + ' )'
            elif restriction["type"] == "max_outdegree":
                query = '''
                    select ?entity  {
                    graph <split:all> {
                        ?entity ?p ?o .
                        }
                        FILTER(isIRI(?entity))
                    } group by ?entity
                    having (count(?entity) > ''' + str(restriction["maximum"]) + ' )'
            else:
                type = restriction["type"]
                raise Exception(f"unknown restriction type: {type}")
            # print(query)
            results = execute_sparql_to_result_silenced(query, sparql_endpoint, sparql_endpoint_options)
            for result in results:
                entity = result.get("entity")
                excluded_entities[restriction["variable"]].add(str(entity))
        filter_string = "\n"
        for variable, entities in excluded_entities.items():
            for entity in entities:
                filter_string += f"FILTER ({variable} != <{entity}>)\n"
        target.mkdir(parents=True, exist_ok=True)
        for split in source.glob("*.sparql"):
            query_unprocessed = split.read_text()
            query_processed = query_unprocessed.replace("### restrictions ###", filter_string, 1)
            target_path = target / split.name
            target_path.write_text(query_processed)


def _load_json_from_file(the_path: Path) -> Any:
    with open(the_path, "rt") as the_file:
        return json.load(the_file)


def _dump_json_to_file(the_path: Path, json_object: Any) -> None:
    with open(the_path, "wt") as the_file:
        return json.dump(json_object, the_file)


def sample_queries(
    dataset: Dataset,
    sparql_endpoint: str,
    sparql_endpoint_options: Optional[Dict[str, Any]] = None,
    continue_on_error: bool = False,
    shuffling_random_seed: int = 58148615010101
) -> None:
    """
    Perform the higher order queries in all subdirectories and store their results in CSV files.

    The output of these higher order queries is such that they are themselves queries.
    to achieve this, the variables of the higher order queries must be as follows:

    * ?sN the subject of the Nth triple in the query
    * ?pN the predicate of the Nth triple
    * ?oN the (entity) object of the Nth triple
    * ?olN the literal object of the Nth triple - only one of olN and oN can exist
    * ?qrNiM the Mth qualifier relation of the Nth triple
    * ?qvNiM the Mth (entity) qualifier value of the Nth triple
    * ?qvlNiM the Mth (literal) qualifier value of the Nth triple - only one of qvNiM and qvlNiM can exist
    * ?diameter The diameter of the query
    * ?XN_targets the (entity) answers to the query, X indicates whether this is a subject, predicate, object, qualifier relation or value, N is the index of the triple
        * If this is a variable used in multiple triples, it is used as a normal one, but with _targets appended.
        * For example in a 2i query, the target would be ?o1_o2_targets
    * ?XN_ltargets like XN_targets but for literal answers. Only one of XN_targets and XN_ltargets can be used.

    Known values in the query must be their URL. The N-th variable (all indices 0 indexed!) must be indicated with "?varN".
    For the indices on the qualifiers, N refers to the triple it belongs to and M is the index of the qulifier on that triple
    Columns which always have the same values MUST be joined by joining the column names by _
    Columns which represent the same variable MUST be joined by joining the column names by _
    Note that this joining rule *also* applies if the common value is between a qv and a subject or object of a triple with more information about the qv.
    These joins are used to verify that the graph is connected.

    For example, a 2 hop query with 1 qualifier on each edge would heave the following variables:

    s0,p0,o0_s1_var,qr0i0,qv0i0,p1,qr1i0,qv1i0,diameter,o1_targets


    In the columns is either a URI, a or multiple URIs separated by "|" for the targets
    In literal columns, there is either a literal, or multiple literals separated by "|" Note that this means literals must not contain the "|" symbol

    For this concrete query, with answers wd:Q4 and wd:Q6
    select ?target
    <<wd:Q1 p:P1 ?var1>> p:P2 wd:Q3
    <<?var1 p:P2 ?target>> p:P3 wd:Q5

    The data in the CSV would be

    wd:Q1,p:P1,?var1,p:P2,wd:Q3,p:P2,p:P3,wd:Q5,2,wd:Q4|wd:Q6

    (note: prefixes should be expanded.)

    The hierarchical structure will be preserved.


    Parameters
    ----------
    dataset: Dataset
        The name of the dataset containing the queries to be executed in its `formulas` directory.
        All formulas (in files ending in .sparql) recursively within this directory will be executed.
        The outcomes will be stored in the datset directory, in the `queries` directory.
        The queries are stored in the same relative path where the .sparql file was found.
        These queries are stored as a gzipped .csv file

    sparql_endpoint: str
        The address of the sparql endpoint

    sparql_endpoint_options: Optional[Dict[str, Any]] = None
        Options to be passed to the SPARQL endpoint upon connecting. This could include things like authentication info, etc.

    continue_on_error: bool = False
        If a query fails and continue_on_error is True, this prints a stacktrace and continues with other queries.
        If a query fails and continue_on_error is False, an Exception is raised.

    """
    # first results are stored in temp/ After splitting hard and easy answers, they are put into the dataset
    formulas_directory: Path = dataset.formulas_location()
    target_directory: Path = dataset.raw_query_csv_location()
    sparql_endpoint_options = sparql_endpoint_options or {}

    if not list(formulas_directory.rglob("*.sparql")):
        logger.warning(f"Empty source directory: {formulas_directory.as_uri()}")

    for query_file_path in formulas_directory.rglob("*.sparql"):
        query = query_file_path.read_text()
        new_query_hash = hashlib.md5(query.encode("utf8")).hexdigest()

        name_stem = query_file_path.stem
        # TODO: Code duplication to converter.py

        # get absolute source path
        relative_source_path = query_file_path.relative_to(formulas_directory)
        relative_source_directory = relative_source_path.parent

        # compute the destination path
        absolute_target_directory = target_directory.joinpath(relative_source_directory)
        absolute_target_directory.mkdir(parents=True, exist_ok=True)
        suffix = ".csv.gz"
        absolute_target_path = absolute_target_directory.joinpath(name_stem).with_suffix(suffix).resolve()
        logger.info(f"{query_file_path.as_uri()} -> {absolute_target_path.as_uri()}")

        # the status file
        status_file_path = absolute_target_directory.joinpath(name_stem + "_stats").with_suffix(".json")

        # we only need to do the querying if the query has updated or there is no such file
        if absolute_target_path.is_file():
            # check the old hash from the stats file if it exists
            if status_file_path.is_file():
                old_stats = _load_json_from_file(status_file_path)
                old_query_hash = old_stats["hash"]
                if old_query_hash == new_query_hash:
                    logger.info(f"Queries already exist for {query_file_path.as_uri()} and hash matches. Not performing the query.")
                    continue
                else:
                    logger.warning(f"Queries exist for {query_file_path.as_uri()}, but hash does not match. Removing and regenerating!")
            else:
                logger.warning(f"Queries exist for {query_file_path.as_uri()}, but no stats file. Removing and regenerating!")
            # an old version exists but hashes do not match, we remove it, the user was warned above
            absolute_target_path.unlink()

        # perform the query and store the results
        logger.info(f"Performing query for {query_file_path.as_uri()} to {absolute_target_path.as_uri()}")
        try:
            try:
                count = _execute_one_query(query, absolute_target_path, sparql_endpoint, sparql_endpoint_options, shuffling_random_seed)
            except HTTPError as e:
                raise Exception(str(e) + str(e.response.content)) from e
            new_stats = {"name": name_stem, "hash": new_query_hash, "raw-count": count}
            try:
                _dump_json_to_file(status_file_path, new_stats)
            except Exception:
                # something went wrong writing the stats file, best to remove it and crash.
                logger.error("Failed writing the stats, removing the file to avoid inconsistent state")
                if status_file_path.exists():
                    status_file_path.unlink()
                raise
        except Exception as err:
            logging.error(f"Something went wrong executing the query in {query_file_path}, removing the output file")
            if absolute_target_path.exists():
                absolute_target_path.unlink()
            if continue_on_error:
                traceback.print_tb(err.__traceback__)
            else:
                raise
    _separate_hard_and_easy_targets(target_directory, dataset.query_csv_location())


def _execute_one_query(query: str, destination_path: Path, sparql_endpoint: str, sparql_endpoint_options: Dict[str, Any], shuffling_random_seed: int) -> int:
    """
    Performs the query provided and writes the results as a CSV to the destination.
    the queries are shuffled randomly (fixed seed) before storing to make later sampling a top-k operation instead of real sampling
    """
    result = execute_csv_sparql_silenced(query, sparql_endpoint, sparql_endpoint_options)
    # convert to string and take of the leading '?'
    assert hasattr(result, "vars") and result.vars is not None, f"No variables for formula query, got: {result}"
    vars: List[Variable] = result.vars

    fieldnames: List[str] = [var.toPython()[1:] for var in vars]  # type: ignore
    assert_query_validity(fieldnames)
    all_queries = list(result)
    query_count = len(all_queries)

    # shuffle all the answers
    r = random.Random(shuffling_random_seed)
    r.shuffle(all_queries)
    with ExitStack() as stack:
        output_file = stack.enter_context(gzip.open(destination_path, compresslevel=6, mode="wt", newline=""))
        writer = csv.DictWriter(output_file, fieldnames, extrasaction="raise", dialect="unix", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for one_query in all_queries:
            writer.writerow(one_query.asdict())
    return query_count


def _separate_hard_and_easy_targets(raw_query_directory: Path, target_path: Path) -> None:
    for (source, target) in pairwise_directories(raw_query_directory, target_path):
        query_sets_found = False
        for file in source.glob("*"):
            if not file.is_file():
                continue
            if file.name.endswith("train.csv.gz"):
                file_prefix = file.name[:-len("train.csv.gz")]
                query_sets_found = True
                break
        if not query_sets_found:
            continue
        train_input_file = source / (file_prefix + "train.csv.gz")
        validation_input_file = source / (file_prefix + "validation.csv.gz")
        test_input_file = source / (file_prefix + "test.csv.gz")
        assert train_input_file.exists() and validation_input_file.exists() and test_input_file.exists(),  \
            f"Found one of train, validation, or test with prefix {file_prefix}, but not all three in {source.relative_to(raw_query_directory)}. Cannot continue"
        # At this point we know that all three input files exist, loading their stats
        train_stats = _load_json_from_file(source / (file_prefix + "train_stats.json"))
        validation_stats = _load_json_from_file(source / (file_prefix + "validation_stats.json"))
        test_stats = _load_json_from_file(source / (file_prefix + "test_stats.json"))

        target.mkdir(parents=True, exist_ok=True)

        train_output_file = target / (file_prefix + "train.csv")
        validation_output_file = target / (file_prefix + "validation.csv")
        test_output_file = target / (file_prefix + "test.csv")

        # for train, we have to rename the target columns to targets-hard and add an empty target-easy column
        train, target_column = _read_csv_and_target_column(train_input_file)
        train["targets-easy"] = ""
        train[target_column + "-hard"] = train[target_column]
        train.pop(target_column)
        _write_dataframe_to_compressed_csv(train, train_output_file)
        train_stats["count"] = train_stats["raw-count"]
        _dump_json_to_file(target / (file_prefix + "train_stats.json"), train_stats)

        # load the datasets
        train, target_column = _read_csv_with_target_as_set(train_input_file)
        validation, _ = _read_csv_with_target_as_set(validation_input_file)
        test, _ = _read_csv_with_target_as_set(test_input_file)
        common_columns = [column_name for column_name in train.columns if not column_name == target_column]

        # create the validation set by filtering stuff from train out
        validation_with_easy_and_hard = _combine_train_validation_answers(train, validation, target_column, common_columns)
        combined_hash = hashlib.md5((train_stats["hash"] + "|" + validation_stats["hash"]).encode("utf8")).hexdigest()
        stats_output = target / (file_prefix + "validation_stats.json")
        _postprocess(validation_with_easy_and_hard, target_column, validation_output_file, validation_stats, combined_hash, stats_output)

        # We will reuse the implementation for train and validation for the test set as well.
        # To do this, we combine train and validation in one dataframe and treat it as train and treat test as validation
        train_validation = pd.merge(train, validation, how="outer", on=common_columns, suffixes=["_train", "_validation"])
        # now we merge the answers
        train_validation[target_column] = train_validation.apply(
            lambda row:
                (row[target_column + "_train"] if isinstance(row[target_column + "_train"], set) else set())
            .union(row[target_column + "_validation"] if isinstance(row[target_column + "_validation"], set) else set()),
            axis=1
        )
        # remove the tmp columns
        train_validation.pop(target_column + "_train")
        train_validation.pop(target_column + "_validation")
        # reuse the combine implementation, assuming identities
        test_with_easy_and_hard = _combine_train_validation_answers(train=train_validation, validation=test, target_column=target_column, common_columns=common_columns)
        combined_hash = hashlib.md5((train_stats["hash"] + "|" + validation_stats["hash"] + "|" + test_stats["hash"]).encode("utf8")).hexdigest()
        stats_output = target / (file_prefix + "test_stats.json")
        _postprocess(test_with_easy_and_hard, target_column, test_output_file, test_stats, combined_hash, stats_output)
        logger.info(f"Done converting from {source} to {target}.")


def _write_dataframe_to_compressed_csv(dataframe: pd.DataFrame, file_without_gz: Path) -> None:
    name = file_without_gz.name
    with_gz = name + ".gz"
    file_with_gz = file_without_gz.resolve().parent / with_gz
    dataframe.to_csv(file_with_gz, compression="gzip")


def _postprocess(queries_with_easy_and_hard: pd.DataFrame, target_column: str, output_file: Path, query_stats: Any, new_hash: str, stats_output: Path) -> None:
    # now we have to convert back from sets to string to be able to write the CSV
    _deterministically_convert_set_column_to_bar_separated(queries_with_easy_and_hard, target_column + "-hard", False)
    _deterministically_convert_set_column_to_bar_separated(queries_with_easy_and_hard, "targets-easy", True)

    _write_dataframe_to_compressed_csv(queries_with_easy_and_hard, output_file)

    # write the stats file with modified hash
    query_stats["hash"] = new_hash
    query_stats["count"] = queries_with_easy_and_hard.shape[0]
    _dump_json_to_file(stats_output, query_stats)


def _deterministically_convert_set_column_to_bar_separated(frame: pd.DataFrame, column_name: str, nan_to_empty: bool) -> None:
    if not nan_to_empty:
        frame[column_name] = frame[column_name].map(lambda the_set: "|".join(sorted(the_set)))
    else:
        frame[column_name] = frame[column_name].map(lambda the_set: "|".join(sorted(the_set)) if isinstance(the_set, set) else "")


def _combine_train_validation_answers(train: pd.DataFrame, validation: pd.DataFrame, target_column: str, common_columns: List[str]) -> pd.DataFrame:
    assert train.shape[0] > 0
    assert validation.shape[0] > 0

    # we have to split the answer sets for validation in the original target and easy_target (the one also in train)
    # we use a inner merge to only get the rows which are in both train and validation
    in_common = pd.merge(train, validation, how="inner", on=common_columns, suffixes=["_train", "_validation"])
    in_common["targets-easy"] = in_common[target_column + "_train"]

    # TODO remove the -hard suffix, this is not needed in the end
    hard_answers = in_common.apply(
        lambda row: row[target_column + "_validation"] - (row[target_column + "_train"]),
        axis=1
    )
    if hard_answers.size > 0:
        in_common[target_column + "-hard"] = hard_answers
    else:
        in_common[target_column + "-hard"] = pd.Index([])
    # get rid of the tmp columns
    in_common.pop(target_column + "_validation")
    in_common.pop(target_column + "_train")

    # Now we merge again to find out which rows have only hard answers and were not in common at all
    merged = pd.merge(validation, in_common, how="left", on=common_columns, suffixes=["_validation-only", None])
    to_be_overwritten = merged[target_column + "-hard"].isnull() & merged["targets-easy"].isnull()
    merged.loc[to_be_overwritten, target_column + "-hard"] = merged.loc[to_be_overwritten, target_column]

    # get rid of the tmp column
    merged.pop(target_column)

    def f(row: pd.Series) -> bool:
        hard_is_empty = len(row[target_column + "-hard"]) == 0
        easy_has_stuff = len(row["targets-easy"]) > 0 if isinstance(row["targets-easy"], set) else False
        result = hard_is_empty and easy_has_stuff
        return result

    # Now, we still have rows with easy targets, but no hard targets, these rows need to be removed
    to_be_dropped = merged.apply(
        # Note: we would shortcut the need to check for Nan in the easy column here, if there are no hard ones there will always be easy ones
        # however, pandas & does not shortcut. the isinstance is actually checking for Nan
        f,
        axis=1
    )
    merged = merged[~to_be_dropped]

    return merged


def _read_csv_and_target_column(input_file: Path) -> Tuple[pd.DataFrame, str]:
    """Read a CSV file into a pandas df. Then converts the column which contains the targets into set objects. Returns the new df and the target column name"""
    dataset = pd.read_csv(input_file, compression="gzip")
    target_columns = [column_name for column_name in dataset.columns if column_name.endswith("_targets")]
    assert len(target_columns) == 1
    target_column = target_columns[0]
    return dataset, target_column


def _read_csv_with_target_as_set(input_file: Path) -> Tuple[pd.DataFrame, str]:
    """Read a CSV file into a pandas df. Then converts the column which contains the targets into set objects. Returns the new df and the target column name"""
    dataset, target_column = _read_csv_and_target_column(input_file)
    dataset[target_column] = dataset[target_column].map(
        lambda value: set(value.split("|"))
    )
    return dataset, target_column


def remove_queries(dataset: Dataset) -> None:
    # we remove both the raw and corrected queries
    shutil.rmtree(dataset.raw_query_csv_location(), ignore_errors=True)
    shutil.rmtree(dataset.query_csv_location(), ignore_errors=True)


subject_matcher = re.compile("^s[0-9]+$")
predicate_matcher = re.compile("^p[0-9]+$")
entity_object_matcher = re.compile("^o[0-9]+$")
literal_object_matcher = re.compile("^ol[0-9]+$")
qualifier_relation_matcher = re.compile("^qr[0-9]+i[0-9]+$")
entity_qualifier_value_matcher = re.compile("^qv[0-9]+i[0-9]+$")
literal_qualifier_value_matcher = re.compile("^qvl[0-9]+i[0-9]+$")


def assert_query_validity(fieldnames: Iterable[str]) -> bool:
    """
    Some heuristics to check whether the headers make sense for a query.
    This is not exhaustive, some specific cases can still pass this test.
    In particular cases where self loops not connected to the rest of the graph go undetected.
    """
    # We need a modifiable list
    fieldnames = list(fieldnames)
    # quick check for duplicates
    allsubparts = [subpart for field in fieldnames for subpart in field.split("_") if not (subpart == "var" or subpart == "targets")]
    duplicates = set([x for x in allsubparts if allsubparts.count(x) > 1])
    assert len(duplicates) == 0, "The following specifications where found in more than one specification: " + str(duplicates)

    expected_triple_count = sum([1 for fields in fieldnames for subpart in fields.split('_') if subject_matcher.match(subpart)])
    assert expected_triple_count > 0, "At least one triple must be completely specififies with subject, predicate and object"
    expected_qualifier_count = sum([1 for fields in fieldnames for subpart in fields.split('_') if qualifier_relation_matcher.match(subpart)])
    # keep track of what has been found
    target_found = False
    diameter_found = False
    spo_found = [[False, False, False] for i in range(expected_triple_count)]
    # The qualifier relation and value must refer to the same triple, we remmeber the triple number and verify in the end.
    qr_qv_found = [[-1, -1] for i in range(expected_qualifier_count)]
    for part in fieldnames:
        # Note: at this point, targets are not yet split into hard and easy!
        if part.endswith("_targets"):
            assert target_found is False, "more than one column with _target found. Giving up."
            target_found = True
            targetHeader = part[:-len("_targets")]
            subparts = targetHeader.split('_')
            assert "var" not in subparts, "'var' cannot occur in the same header with 'target', , got {part}"
        if part.endswith("_var"):
            varheader = part[:-len("_var")]
            subparts = varheader.split("_")
            assert "targets" not in subparts, f"'var' cannot occur in the same header with 'target', got {part}"
        if part.endswith("_targets") or part.endswith("_var"):
            # This header must contain only s, o, qr, qv
            for subpart in subparts:
                if not any((
                    subject_matcher.match(subpart),
                    entity_object_matcher.match(subpart),
                    literal_object_matcher.match(subpart),
                    qualifier_relation_matcher.match(subpart),
                    literal_qualifier_value_matcher.match(subpart),
                    entity_qualifier_value_matcher.match(subpart),
                )):
                    raise ValueError(
                        f"the target header can only contain subject, predicate, object, query relation, or query "
                        f"value, contained {subpart}",
                    )
        else:
            # split the part if it is used multiple times
            subparts = part.split("_")
        assert not len(subparts) == 0, "Did you create a column with a header without any actual fields in it?"
        if len(subparts) == 1:
            if subparts[0] == "diameter":
                assert not diameter_found, "diameter specified more than once"
                diameter_found = True
                continue
        elif len(subparts) > 1:
            # there can be no predicates, qualifier_relations and diameter here
            for subpart in subparts:
                if subpart.startswith('p') or subpart.startswith('qr') or subpart == 'diameter':
                    logging.warning(f"""Found a joined header {part} with a part that is typically not joined {subpart}.
                    This might be intended, but is strange. Perhaps a case  where you want to do something with a shared edge?""")
        for subpart in subparts:
            # we treat each different: subject, predicate, object, qr, qv
            if subject_matcher.match(subpart):
                # it is a subject
                tripleIndex = int(subpart[1:])
                assert tripleIndex < expected_triple_count, \
                    f"Found a {subpart} refering to triple {tripleIndex} while we only have {expected_triple_count} triples"
                assert not spo_found[tripleIndex][0]
                spo_found[tripleIndex][0] = True
            elif predicate_matcher.match(subpart):
                tripleIndex = int(subpart[1:])
                assert tripleIndex < expected_triple_count, \
                    f"Found a {subpart} refering to triple {tripleIndex} while we only have {expected_triple_count} triples"
                assert not spo_found[tripleIndex][1]
                spo_found[tripleIndex][1] = True
            elif entity_object_matcher.match(subpart) or literal_object_matcher.match(subpart):
                tripleIndex = int(subpart[1:]) if entity_object_matcher.match(subpart) else int(subpart[2:])
                assert tripleIndex < expected_triple_count, \
                    f"Found a {subpart} refering to triple {tripleIndex} while we only have {expected_triple_count} triples"
                assert not spo_found[tripleIndex][2]
                spo_found[tripleIndex][2] = True
            elif qualifier_relation_matcher.match(subpart):
                indices = subpart[2:].split('i')
                tripleIndex = int(indices[0])
                assert tripleIndex < expected_triple_count, f"qualifier relation {subpart} refers to non existing triple {tripleIndex}"
                qualIndex = int(indices[1])
                assert qr_qv_found[qualIndex][0] == -1, f"qualifier relation qrXi{qualIndex} set twice"
                qr_qv_found[qualIndex][0] = tripleIndex
            elif entity_qualifier_value_matcher.match(subpart) or literal_qualifier_value_matcher.match(subpart):
                indices = subpart[2:].split('i') if entity_qualifier_value_matcher.match(subpart) else subpart[3:].split('i')
                tripleIndex = int(indices[0])
                assert tripleIndex < expected_triple_count, f"qualifier value {subpart} refers to non existing triple {tripleIndex}"
                qualIndex = int(indices[1])
                assert qualIndex < expected_qualifier_count, f"Found qualifier value {subpart} for which there is no corresponding qr{qualIndex}"
                assert qr_qv_found[qualIndex][1] == -1, f"qualifier value qvXi{qualIndex} set twice"
                qr_qv_found[qualIndex][1] = tripleIndex
            else:
                raise AssertionError(f"Unknown column with name '{subpart}'")
    # The qualifier relation and value must refer to the same triple, we remmeber the triple number and verify in the end.
    assert target_found, "No column found with _target"
    assert diameter_found, "No query diameter specified"
    for (index, spo) in enumerate(spo_found):
        assert spo[0], f"Could not find subject s{index}"
        assert spo[1], f"Could not find predicate p{index}"
        assert spo[2], f"Could not find object o{index}"
    for (index, qr_qv) in enumerate(qr_qv_found):
        assert qr_qv[0] != -1, f"Could not find qrXi{index}"
        assert qr_qv[1] != -1, f"Could not find qvXi{index}"
    # Finally, some more checks to catch at least some cases where the query graph is not connected. For a one triple query we are done here
    if expected_triple_count == 1:
        return True
    # For each triple there must be at least one end that is joined with an s or o of another triple or with a qv.
    # We only check that it at least co-occurs with something. This leaves some case where triples are looping on themselves undiscovered.
    # Note that at this point we are already sure that all qualifiers are connected to triples and all fields have already been checked above.
    # In principle, this loop could occur as part of the above code checking the field, but that would render it pretty much unreadble.
    connectionFound = [False for i in range(expected_triple_count)]
    for part in fieldnames:
        if part.endswith("_targets"):
            targetHeader = part[:-len("_targets")]
            subparts = targetHeader.split('_')
        elif part.endswith("_var"):
            varheader = part[:-len("_var")]
            subparts = varheader.split("_")
        else:
            # split the part if it is used multiple times
            subparts = part.split("_")
        if len(subparts) == 1:
            # single fields do not give us any information about connectedness
            continue
        for subpart in subparts:
            # we treat each different: subject, predicate, object, qr, qv.
            if subject_matcher.match(subpart):
                tripleIndex = int(subpart[1:])
                connectionFound[tripleIndex] = True
            elif entity_object_matcher.match(subpart) or literal_object_matcher.match(subpart):
                tripleIndex = int(subpart[1:])
                connectionFound[tripleIndex] = True
            else:
                # Just fine. Other things could occur in these fields, but we can ignore these here.
                pass
    assert all(connectionFound), "It seems like not all triples in the query are connected to another triple or qualifier"
    return True
