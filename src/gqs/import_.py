"""Import a dataset that was created in the format as for the KGReasoning framework

    This focuses on getting three parts:
    1. the ID mapping
    2. the data splits
    3. the test queries.

    It does not convert the train and validation queries
      because it is up to any system to create them the way they want, potentially using the gqs framework.
"""


import logging
import pickle
import pathlib
from typing import Any, Callable, Type, TypeVar, cast
from gqs.conversion import protobuf_builder
from gqs.dataset import Dataset
from gqs.mapping import RelationMapper, EntityMapper
from gqs.conversion import QueryBuilder
from gqs.query_representation.query_pb2 import Query, QueryData

logger = logging.getLogger(__name__)


def KGReasoning_to_zero_qual_queries_dataset(import_source: pathlib.Path, dataset: Dataset, lenient: bool) -> None:
    dataset.location().mkdir()
    dataset.mapping_location().mkdir()
    _convert_mapper(import_source / "id2ent.pkl", dataset.entity_mapping_location())
    _convert_mapper(import_source / "id2rel.pkl", dataset.relation_mapping_location())
    dataset.splits_location().mkdir(parents=True)
    _convert_graph_splits(import_source, dataset)
    dataset.query_proto_location().mkdir(parents=True)
    _convert_queries(import_source, dataset, lenient)


def _convert_mapper(id2X_file: pathlib.Path, target_file: pathlib.Path) -> None:
    with open(id2X_file, "rb") as f:
        mapping = pickle.load(f)
    # sanity checks
    num_ids = len(mapping)
    for i in range(num_ids):
        assert i in mapping, f"The id {i} was not found in the mapping file {id2X_file}. Cannot convert"
    with open(target_file, "w") as output:
        sep = ""
        for i in range(num_ids):
            output.write(f"{sep}{mapping[i]}")
            sep = "\n"


def _convert_graph_splits(import_source: pathlib.Path, dataset: Dataset) -> None:
    ent_map: EntityMapper = dataset.entity_mapper
    rel_map: RelationMapper = dataset.relation_mapper

    for source_name, target_name in [("test.txt", "test.nt"), ("valid.txt", "validation.nt"), ("train.txt", "train.nt")]:
        with open(import_source / source_name) as input_file:
            with open(dataset.splits_location() / target_name, "w") as output_file:
                for line in input_file:
                    parts = line.split()
                    assert len(parts) == 3, f"splitting the line {line} in {source_name} did not split in 3 parts"
                    try:
                        s_int = int(parts[0])
                        p_int = int(parts[1])
                        o_int = int(parts[2])
                    except ValueError as e:
                        raise Exception("for the line {line} in {source_name}, one of the parts could not be parsed as an int") from e
                    s = ent_map.inverse_lookup(s_int)
                    p = rel_map.inverse_lookup(p_int)
                    o = ent_map.inverse_lookup(o_int)
                    line = f"<{s}> <{p}> <{o}> .\n"
                    output_file.write(line)


T = TypeVar('T')

KGQueryShape = tuple[Any, ...]
KGQueryInstance = tuple[Any, ...]


def _mappers(rel_map: RelationMapper, ent_map: EntityMapper, builder_factory: Type[QueryBuilder[T]]) -> dict[KGQueryShape, tuple[str, Callable[[KGQueryInstance], QueryBuilder[T]]]]:
    # In all of these, we map from int to str. Most builders will map this back to int. this could be optimized out.

    def _1hop(KGlib_query: KGQueryInstance) -> QueryBuilder[T]:
        builder = builder_factory(1, 0)
        builder.set_diameter(1)
        (e, (r,)) = KGlib_query
        # most builders will map this back..
        builder.set_subject_predicate_entity_object(0,
                                                    ent_map.inverse_lookup(e),
                                                    rel_map.inverse_lookup(r),
                                                    EntityMapper.get_target_entity_name())
        return builder

    def _2hop(KGlib_query: KGQueryInstance) -> QueryBuilder[T]:
        builder = builder_factory(2, 0)
        builder.set_diameter(2)
        (e, (r0, r1)) = KGlib_query
        # most builders will map this back..
        builder.set_subject_predicate_entity_object(0,
                                                    ent_map.inverse_lookup(e),
                                                    rel_map.inverse_lookup(r0),
                                                    "?var0")
        builder.set_subject_predicate_entity_object(1,
                                                    "?var0",
                                                    rel_map.inverse_lookup(r1),
                                                    EntityMapper.get_target_entity_name())
        return builder

    def _3hop(KGlib_query: KGQueryInstance) -> QueryBuilder[T]:
        builder = builder_factory(3, 0)
        builder.set_diameter(3)
        e, (r0, r1, r2) = KGlib_query
        # most builders will map this back..
        builder.set_subject_predicate_entity_object(0,
                                                    ent_map.inverse_lookup(e),
                                                    rel_map.inverse_lookup(r0),
                                                    "?var0")
        builder.set_subject_predicate_entity_object(1,
                                                    "?var0",
                                                    rel_map.inverse_lookup(r1),
                                                    "?var1")
        builder.set_subject_predicate_entity_object(2,
                                                    "?var1",
                                                    rel_map.inverse_lookup(r2),
                                                    EntityMapper.get_target_entity_name())
        return builder

    def _2i(KGlib_query: KGQueryInstance) -> QueryBuilder[T]:
        builder = builder_factory(2, 0)
        builder.set_diameter(1)
        ((e0, (r0,)), (e1, (r1,))) = KGlib_query
        # most builders will map this back..
        builder.set_subject_predicate_entity_object(0,
                                                    ent_map.inverse_lookup(e0),
                                                    rel_map.inverse_lookup(r0),
                                                    EntityMapper.get_target_entity_name())
        builder.set_subject_predicate_entity_object(1,
                                                    ent_map.inverse_lookup(e1),
                                                    rel_map.inverse_lookup(r1),
                                                    EntityMapper.get_target_entity_name())
        return builder

    def _3i(KGlib_query: KGQueryInstance) -> QueryBuilder[T]:
        builder = builder_factory(3, 0)
        builder.set_diameter(1)
        ((e0, (r0,)), (e1, (r1,)), (e2, (r2,))) = KGlib_query
        # most builders will map this back..
        builder.set_subject_predicate_entity_object(0,
                                                    ent_map.inverse_lookup(e0),
                                                    rel_map.inverse_lookup(r0),
                                                    EntityMapper.get_target_entity_name())
        builder.set_subject_predicate_entity_object(1,
                                                    ent_map.inverse_lookup(e1),
                                                    rel_map.inverse_lookup(r1),
                                                    EntityMapper.get_target_entity_name())
        builder.set_subject_predicate_entity_object(2,
                                                    ent_map.inverse_lookup(e2),
                                                    rel_map.inverse_lookup(r2),
                                                    EntityMapper.get_target_entity_name())

        return builder

    def _2i_1hop(KGlib_query: KGQueryInstance) -> QueryBuilder[T]:
        builder = builder_factory(3, 0)
        builder.set_diameter(2)
        ((e0, (r0,)), (e1, (r1,))), (r2,) = KGlib_query
        # most builders will map this back..
        builder.set_subject_predicate_entity_object(0,
                                                    ent_map.inverse_lookup(e0),
                                                    rel_map.inverse_lookup(r0),
                                                    "?var0")
        builder.set_subject_predicate_entity_object(1,
                                                    ent_map.inverse_lookup(e1),
                                                    rel_map.inverse_lookup(r1),
                                                    "?var0")
        builder.set_subject_predicate_entity_object(2,
                                                    "?var0",
                                                    rel_map.inverse_lookup(r2),
                                                    EntityMapper.get_target_entity_name())
        return builder

    def _1hop_2i(KGlib_query: KGQueryInstance) -> QueryBuilder[T]:
        builder = builder_factory(3, 0)
        builder.set_diameter(2)
        ((e0, (r0, r1)), (e1, (r2,))) = KGlib_query
        # most builders will map this back..
        builder.set_subject_predicate_entity_object(0,
                                                    ent_map.inverse_lookup(e0),
                                                    rel_map.inverse_lookup(r0),
                                                    "?var0")
        builder.set_subject_predicate_entity_object(1,
                                                    "?var0",
                                                    rel_map.inverse_lookup(r1),
                                                    EntityMapper.get_target_entity_name())
        builder.set_subject_predicate_entity_object(2,
                                                    ent_map.inverse_lookup(e1),
                                                    rel_map.inverse_lookup(r2),
                                                    EntityMapper.get_target_entity_name())

        return builder

    mapping: dict[KGQueryShape, tuple[str, Callable[[KGQueryInstance], QueryBuilder[T]]]] = {
        ('e', ('r',)): ("1hop", _1hop),
        ('e', ('r', 'r')): ("2hop", _2hop),
        ('e', ('r', 'r', 'r')): ("3hop", _3hop),
        (('e', ('r',)), ('e', ('r',))): ("2i", _2i),
        (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): ("3i", _3i),
        ((('e', ('r',)), ('e', ('r',))), ('r',)): ('2i-1hop', _2i_1hop),
        (('e', ('r', 'r')), ('e', ('r',))): ('1hop-2i', _1hop_2i)
    }
    return mapping


def _convert_queries(import_source: pathlib.Path, dataset: Dataset, lenient: bool) -> None:
    """Convert the queries to the gqs format

    Args:
        import_source (pathlib.Path): the folder containing the queries to be imported
        dataset (Dataset): the target dataset
        lenient (bool): if false, raises an exception if an unknown query shape is encountered, otherwise logs a warning

    Raises:
        Exception: raises if an unknown query shape is encountered and lenient is false
    """
    builder_factory = protobuf_builder(dataset.relation_mapper, dataset.entity_mapper)
    mappers = _mappers(dataset.relation_mapper, dataset.entity_mapper, builder_factory)
    with open(import_source / "test-queries.pkl", "rb") as f:
        queries = pickle.load(f)
    with open(import_source / "test-easy-answers.pkl", "rb") as f:
        all_easy_answers = pickle.load(f)
    with open(import_source / "test-hard-answers.pkl", "rb") as f:
        all_hard_answers = pickle.load(f)

    for query_shape, query_instances in queries.items():
        query_shape = cast(KGQueryShape, query_shape)
        query_instances = cast(list[KGQueryInstance], query_instances)
        if query_shape not in mappers:
            if lenient:
                logger.warning(f"The shape {query_shape} was not found. Likely not yet implemented")
                continue
            else:
                raise Exception(f"The shape {query_shape} was not found. Likely not yet implemented")
        query_shape_name, mapper = mappers[query_shape]
        proto_query_data = QueryData()
        for query in query_instances:
            builder = mapper(query)
            easy_answers = all_easy_answers[query]
            builder.set_easy_entity_targets([dataset.entity_mapper.inverse_lookup(t) for t in easy_answers])
            hard_answers = all_hard_answers[query]
            builder.set_hard_entity_targets([dataset.entity_mapper.inverse_lookup(t) for t in hard_answers])
            proto_query: Query = builder.build()
            proto_query_data.queries.append(proto_query)

        output_folder = dataset.query_proto_location() / query_shape_name / "0qual"
        output_folder.mkdir(parents=True)
        output_file_name = output_folder / "test.proto"
        with open(output_file_name, "wb") as output_file:
            output_file.write(proto_query_data.SerializeToString())
        print(f"Done with shape {query_shape}")
