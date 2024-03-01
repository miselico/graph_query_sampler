"""Export th queries into a format compatible with some query answering frameworks like KGReasoning
This code is inspired by an notebook for conversion by Max Zwager
"""

import logging
import pickle
import pathlib
from typing import Any, Callable, List, Tuple
import pandas as pd
from gqs.dataset import Dataset
from gqs.mapping import RelationMapper, EntityMapper

logger = logging.getLogger(__name__)

"""These mapping are in accordance with the example mappings in sesources/formulas_example
They map a formula to its name in the KGreasoning framework and a transformation function.
"""


def _get_transforms(rmap: RelationMapper, emap: EntityMapper) -> dict[str, Tuple[Tuple[Any, ...], Callable[[pd.Series], Tuple[Any, ...]]]]:
    _transforms: dict[str, Tuple[Tuple[Any, ...], Callable[[pd.Series], Tuple[Any, ...]]]] = {
        '1hop': (('e', ('r',)), lambda x: (emap.lookup(x["s0"]), (rmap.lookup(x["p0"]),))),
        '2hop': (('e', ('r', 'r')), lambda x: (emap.lookup(x["s0"]), (rmap.lookup(x["p0"]), rmap.lookup(x["p1"])))),
        '3hop': (('e', ('r', 'r', 'r')), lambda x: (emap.lookup(x["s0"]), (rmap.lookup(x["p0"]), rmap.lookup(x["p1"]), rmap.lookup(x["p2"])))),
        '2i': ((('e', ('r',)), ('e', ('r',))), lambda x: ((emap.lookup(x["s0"]), (rmap.lookup(x["p0"]),)), (emap.lookup(x["s1"]), (rmap.lookup(x["p1"]),)))),
        '3i': ((('e', ('r',)), ('e', ('r',)), ('e', ('r',))), lambda x: ((emap.lookup(x["s0"]), (rmap.lookup(x["p0"]),)), (emap.lookup(x["s1"]), (rmap.lookup(x["p1"]),)), (emap.lookup(x["s2"]), (rmap.lookup(x["p2"]),)))),
        '2i-1hop': (((('e', ('r',)), ('e', ('r',))), ('r',)), lambda x: (((emap.lookup(x["s0"]), (rmap.lookup(x["p0"]),)), (emap.lookup(x["s1"]), (rmap.lookup(x["p1"]),))), (rmap.lookup(x["p2"]),))),
        '1hop-2i': ((('e', ('r', 'r')), ('e', ('r',))), lambda x: ((emap.lookup(x["s2"]), (rmap.lookup(x["p2"]), rmap.lookup(x["p0"]))), (emap.lookup(x["s1"]), (rmap.lookup(x["p1"]),)))),
        # '2in': lambda x:  ((x[0], (x[1],)), (x[2], (x[3], x[4]))),
        # '3in': lambda x:  ((x[0], (x[1],)), (x[2], (x[3],)), (x[4], (x[5], x[6]))),
        # 'inp': lambda x:  (((x[0], (x[1],)), (x[2], (x[3], x[4]))), (x[5],)),
        # 'pin': lambda x:  ((x[0], (x[1], x[2])), (x[3], (x[4], x[5]))),
        # 'pni': lambda x:  ((x[0], (x[1], x[2], x[3])), (x[4], (x[5],))),
        # '2u-DNF': lambda x:  ((x[0], (x[1],)), (x[2], (x[3],)), (x[4],)),
        # 'up-DNF': lambda x:  (((x[0], (x[1],)), (x[2], (x[3],)), (x[4],)), (x[5],)),
        # '2u-DM': lambda x:  (((x[0], (x[1], x[2])), (x[3], (x[4], x[5]))), (x[6],)),
        # 'up-DM': lambda x:  (((x[0], (x[1], x[2])), (x[3], (x[4], x[5]))), (x[6], x[7])),
    }
    return _transforms


def zero_qual_queries_dataset_to_KGReasoning(dataset: Dataset) -> None:
    """
    Transforms a StarQE CSV file into a dictionary of tuples structures according to KGReasoning framework.

    Note that this assumes the queries to be in /datasets/{datasetname}/queries/csv/{formula}/0qual/[test,train,validation].csv.gz

    Args:
        dataset (Dataset): the dataset
    """
    relmap, entmap = dataset.get_mappers()
    transforms = _get_transforms(relmap, entmap)
    for pattern_path in dataset.query_csv_location().glob('*'):
        assert pattern_path.name in transforms, "There are patterns in the folder to be converted that are not known to the exporter. Aborting."
    for split, kgreasoning_name in [("train", "train"), ("validation", "valid"), ("test", "test")]:
        queries_output_location = dataset.export_kgreasoning_location() / (kgreasoning_name + "-queries.pkl")
        easy_answers_output_location = None if split == "train" else dataset.export_kgreasoning_location() / (kgreasoning_name + "-easy-answers.pkl")
        hard_answers_output_location = dataset.export_kgreasoning_location() / ("train-answers.pkl" if split == "train" else kgreasoning_name + "-hard-answers.pkl")
        all_queries: dict[Tuple[Any, ...], List[Tuple[Any, ...]]] = {}
        all_easy_answers: dict[Tuple[Any, ...], set[int]] = {}
        all_hard_answers: dict[Tuple[Any, ...], set[int]] = {}
        for pattern in transforms.keys():
            input_location = dataset.query_csv_location() / pattern / '0qual' / f'{split}.csv.gz'
            if not input_location.exists():
                logger.warning(f"No queries found for pattern {pattern} in split {split}. Expected {input_location} to exist.")
                continue
            # convert and collect
            kg_reasoning_pattern_name, transform = transforms[pattern]
            new_queries, new_easy, new_hard = _zero_qual_queries_csv_to_KGReasoning(input_location, transform, entmap)

            # remember the queries and the results
            all_queries[kg_reasoning_pattern_name] = new_queries
            all_easy_answers = all_easy_answers | new_easy
            all_hard_answers = all_hard_answers | new_hard
        # write them out:
        queries_output_location.parent.mkdir(parents=True, exist_ok=True)
        with open(queries_output_location, 'wb') as open_query_output:
            pickle.dump(all_queries, open_query_output)
        if easy_answers_output_location is not None:
            with open(easy_answers_output_location, 'wb') as open_easy_answers:
                pickle.dump(all_easy_answers, open_easy_answers)
        with open(hard_answers_output_location, 'wb') as open_hard_answers:
            pickle.dump(all_hard_answers, open_hard_answers)
        # finally, the total number of entities and relations is needed:
        stats_output_location = (dataset.export_kgreasoning_location() / "stats.txt").resolve()

        rel_type_count = relmap.number_of_relation_types()
        entity_count = entmap.number_of_real_entities()
        with open(stats_output_location, "wt") as open_stats_output:
            open_stats_output.write(f"numentity: {entity_count}\nnumrelations: {rel_type_count}")


def _zero_qual_queries_csv_to_KGReasoning(
    input_location: pathlib.Path,
    mapper: Callable[[pd.Series], Tuple[Any, ...]],
    entity_mapper: EntityMapper
) -> tuple[List[Tuple[Any, ...]], dict[Tuple[Any, ...], set[int]], dict[Tuple[Any, ...], set[int]]]:

    df = pd.read_csv(input_location, sep=',', keep_default_na=False)
    # df.drop(    # Drop columns containing variables and diameter
    #     columns=[c for c in df.columns.to_list() if (c.endswith('var') or c.endswith('target'))].append('diameter'),
    #     inplace=True)
    query_list = []
    easy_anwsers: dict[Tuple[Any, ...], set[int]] = dict()
    hard_answers: dict[Tuple[Any, ...], set[int]] = dict()
    for _, row in df.iterrows():
        query = mapper(row)
        easy_answers_for_query = None
        hard_answers_for_query = None
        for c in df.columns:
            if c.endswith('targets-easy'):
                if row[c] == "":
                    easy_answers_for_query = set()
                else:
                    easy_answers_for_query = set([entity_mapper.lookup(entity) for entity in row[c].split("|")])
            elif c.endswith('targets-hard'):
                hard_answers_for_query = set([entity_mapper.lookup(entity) for entity in row[c].split("|")])
        assert easy_answers_for_query is not None and hard_answers_for_query is not None

        query_list.append(query)
        easy_anwsers[query] = easy_answers_for_query
        hard_answers[query] = hard_answers_for_query

    return query_list, easy_anwsers, hard_answers
