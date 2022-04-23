"""Convert text queries to numeric binary."""
import csv
import gzip
import json
import logging
import dill
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (Callable, Generic, Iterable, Optional, Sequence, Tuple,
                    Type, TypeVar)

import torch

from gqs.types import LongTensor

from .mapping import EntityMapper  # , get_entity_mapper, get_relation_mapper
from .mapping import RelationMapper
from .query_represenation.query_pb2 import \
    EntityOrLiteral as pb_EntityOrLiteral
from .query_represenation.query_pb2 import Qualifier as pb_Qualifier
from .query_represenation.query_pb2 import Query as pb_Query
from .query_represenation.query_pb2 import QueryData as pb_QueryData
from .query_represenation.query_pb2 import Triple as pb_Triple
from .query_represenation.torch import TorchQuery

__all__ = [
    "convert_all",
    "protobuf_builder",
    "torch_query_builder"
    # "StrippedSPARQLResultBuilder"
]

logger = logging.getLogger(__name__)

T = TypeVar('T')


class QueryBuilder(Generic[T], ABC):
    @classmethod
    @abstractmethod
    def __init__(self, number_of_triples: int, number_of_qualifiers: int) -> None:
        pass

    @abstractmethod
    def set_subject(self, index: int, entity: str) -> None:
        pass

    @abstractmethod
    def set_entity_object(self, index: int, entity: str) -> None:
        pass

    @abstractmethod
    def set_literal_object(self, index: int, entity: str) -> None:
        pass

    @abstractmethod
    def set_predicate(self, index: int, predicate: str) -> None:
        pass

    def set_subject_predicate_entity_object(self, tripleIndex: int, subject: str, predicate: str, obj: str) -> None:
        """Helper function to set subject, predicate and object at once"""
        self.set_subject(tripleIndex, subject)
        self.set_predicate(tripleIndex, predicate)
        self.set_entity_object(tripleIndex, obj)

    @abstractmethod
    def set_qualifier_rel(self, tripleIndex: int, index: int, predicate: str) -> None:
        pass

    @abstractmethod
    def set_qualifier_entity_val(self, tripleIndex: int, index: int, value: str) -> None:
        pass

    def set_qualifier_rel_entity_val(self, tripleIndex: int, qualifier_index: int, predicate: str, value: str) -> None:
        """
        Set the relation `predicate` and value `value` for the qualifier attached to triple with index `tripleIndex` and qualifier index `qualifier_index`
        """
        self.set_qualifier_rel(tripleIndex, qualifier_index, predicate)
        self.set_qualifier_entity_val(tripleIndex, qualifier_index, value)

    @abstractmethod
    def set_qualifier_literal_val(self, tripleIndex: int, index: int, value: str) -> None:
        pass

    @abstractmethod
    def set_easy_entity_targets(self, values: Iterable[str]) -> None:
        pass

    @abstractmethod
    def set_easy_literal_targets(self, values: Iterable[str]) -> None:
        pass

    @abstractmethod
    def set_hard_entity_targets(self, values: Iterable[str]) -> None:
        pass

    def set_hard_literal_targets(self, values: Iterable[str]) -> None:
        pass

    @abstractmethod
    def set_diameter(self, diameter: int) -> None:
        pass

    @abstractmethod
    def build(self) -> T:
        pass

    @staticmethod
    def get_file_extension() -> str:
        """Get the extension to be used for files stored with the store method. Each deriving class must implement its own static method."""
        raise NotImplementedError("Each class deriving must implement its own version fo this.s")

    @abstractmethod
    def store(self, collection: Iterable[T], absolute_target_path: Path) -> None:
        pass


def _get_triple_and_qualifier_count_from_headers(fieldnames: Sequence[str]) -> Tuple[int, int]:
    all_headers = set([subpart.strip() for header in fieldnames for subpart in header.split("_")])

    # We count the number of subjects in the headers to get the number of triples
    number_of_triples = sum(1 for header in all_headers if re.match(r'^s[0-9]+$', header.strip()))

    # We count the number of query relations in the headers to get the number of triples
    number_of_qualifiers = sum(1 for header in all_headers if re.match(r'^qr[0-9]+i[0-9]+$', header.strip()))
    return number_of_triples, number_of_qualifiers


def convert_one(absolute_source_path: Path, absolute_target_path: Path, builderClass: Type[QueryBuilder[T]]) -> None:
    """Convert a file of textual queries to another format as specified in the builder."""
    converted = []

    with gzip.open(absolute_source_path, mode="rt", newline="") as input_file:
        reader = csv.DictReader(input_file, dialect="unix", quoting=csv.QUOTE_MINIMAL)
        assert reader.fieldnames is not None
        number_of_triples, number_of_qualifiers = _get_triple_and_qualifier_count_from_headers(reader.fieldnames)

        query_count = 0
        for row in reader:
            query_count += 1
            builder = builderClass(number_of_triples, number_of_qualifiers)
            for (part, value) in row.items():
                if part == "":
                    # empty headers are skipped
                    continue
                sub_parts = part.split("_")

                if sub_parts[-1] in {"targets-easy", "targets-hard", "ltargets-easy", "ltargets-hard"}:
                    # store the targets
                    target_spec = sub_parts[-1]
                    easy = target_spec.endswith("easy")
                    literals = target_spec.startswith("ltargets")

                    if easy:
                        assert len(sub_parts) == 1, "In the easy targets column, there can be no specification of the position. This must only be placed on the hard targets column"
                    if value.strip() == "":
                        assert easy, "Only easy targets can be empty"
                        targets = []
                    else:
                        targets = value.split("|")
                    assert len(set(targets)) == len(targets), f"Targets must be unique, got {targets}"

                    if easy and literals:
                        builder.set_easy_literal_targets(targets)
                    elif not easy and literals:
                        builder.set_hard_literal_targets(targets)
                    elif easy and not literals:
                        builder.set_easy_entity_targets(targets)
                    elif not easy and not literals:
                        builder.set_hard_entity_targets(targets)
                    else:
                        raise AssertionError("All cases should be handled, else should never be taken")
                    # we override the value with "TARGET"
                    value = EntityMapper.get_target_entity_name()
                    # chop off the targets
                    sub_parts = sub_parts[:-1]
                elif sub_parts[-1] == "var":
                    assert EntityMapper.is_valid_variable_name(value), f"Invalid variable name, got {value}"
                    # No special action needed except stripping it off, the value of these variables will be enough to do the right thing
                    sub_parts = sub_parts[:-1]
                # var and tatget specifiers are chopped off. Now we still deal with the field itself
                for subpart in sub_parts:
                    subpart = subpart.strip()
                    # we treat each different: subject, predicate, object, qr, qv
                    if subpart.startswith("s"):
                        # it is a subject
                        triple_index = int(subpart[1:])
                        builder.set_subject(triple_index, value)
                    elif subpart.startswith("p"):
                        triple_index = int(subpart[1:])
                        builder.set_predicate(triple_index, value)
                    elif subpart.startswith("o"):
                        triple_index = int(subpart[1:])
                        builder.set_entity_object(triple_index, value)
                    elif subpart.startswith("lo"):
                        triple_index = int(subpart[2:])
                        builder.set_literal_object(triple_index, value)
                    elif subpart.startswith("qr"):
                        indices = subpart[2:].split("i")
                        triple_index = int(indices[0])
                        qualifier_index = int(indices[1])
                        builder.set_qualifier_rel(triple_index, qualifier_index, value)
                    elif subpart.startswith("qv"):
                        indices = subpart[2:].split("i")
                        triple_index = int(indices[0])
                        qualifier_index = int(indices[1])
                        builder.set_qualifier_entity_val(triple_index, qualifier_index, value)
                    elif subpart.startswith("lqv"):
                        indices = subpart[3:].split("i")
                        triple_index = int(indices[0])
                        qualifier_index = int(indices[1])
                        builder.set_qualifier_literal_val(triple_index, qualifier_index, value)
                    elif subpart == "diameter":
                        builder.set_diameter(int(value))
                    else:
                        logger.warning(f"Unknown column with name \"{subpart}\" - ignored")
            converted.append(builder.build())
    # Only storing left
    if query_count == 0:
        logging.warning(f"No triples found in {absolute_source_path}, the binary file will not contain any queries.")
        # need to create the first builder still. Otehrwise we reuse it from the loop
        builder = builderClass(number_of_triples, number_of_qualifiers)
    builder.store(converted, absolute_target_path)


def convert_all(
    source_directory: Path,
    target_directory: Path,
    builder: Type[QueryBuilder[T]],
    filter: Optional[Callable[[str], bool]] = None
) -> None:
    """Convert all textual queries to the format specified by the builder."""
    if not list(source_directory.rglob("*.csv.gz")):
        logger.warning(f"Empty source directory: {source_directory}")
    for query_file_path in source_directory.rglob("*.csv.gz"):
        # take ".csv.gz" of
        name_stem = Path(query_file_path.stem).stem
        if filter and not filter(name_stem):
            continue
        # get absolute source path
        relative_source_path = query_file_path.relative_to(source_directory)
        relative_source_directory = relative_source_path.parent
        absolute_source_directory = source_directory.joinpath(relative_source_directory).resolve()

        # compute the destination path
        absolute_target_directory = target_directory.joinpath(relative_source_directory)
        absolute_target_directory.mkdir(parents=True, exist_ok=True)
        absolute_target_path = absolute_target_directory.joinpath(name_stem).with_suffix(builder.get_file_extension()).resolve()

        logger.info(f"{query_file_path.as_uri()} -> {absolute_target_path.as_uri()}")

        # Read stats
        # TODO: Why do we need this?
        source_stats_file_path = absolute_source_directory.joinpath(name_stem + "_stats").with_suffix(".json").resolve()
        if not source_stats_file_path.is_file():
            raise Exception(f"Stats in {source_stats_file_path} not found")
        with source_stats_file_path.open(mode="r") as stats_file:
            source_stats = json.load(stats_file)

        # check the old hash from the status file
        target_stats_file_path = absolute_target_directory.joinpath(name_stem + "_stats").with_suffix(".json").resolve()

        # we only convert if the there is no such file or the hash has changed (the original query has updated)
        if absolute_target_path.is_file():
            if target_stats_file_path.is_file():
                with target_stats_file_path.open(mode="r") as stats_file:
                    target_stats = json.load(stats_file)
                target_hash = target_stats["hash"]
                source_hash = source_stats["hash"]
                if target_hash == source_hash:
                    logger.info(f"Queries already converted for {query_file_path.as_uri()} and hash matches. Not converting again.")
                    continue
                else:
                    # an old version exists but hashes do not match, we remove it, warn the user
                    logger.warning(f"Queries exist for {query_file_path.as_uri()}, but hash does not match. Removing and regenerating!")
            else:
                logger.warning(f"Queries exist for {query_file_path.as_uri()}, but no stats found. Removing and regenerating!")
            absolute_target_path.unlink()

        # perform the conversion and store the results
        logger.info(f"Performing conversion from {query_file_path.as_uri()} to {absolute_target_path.as_uri()}")
        try:
            convert_one(query_file_path, absolute_target_path, builder)
            # augment the stats with succesful conversion information
            source_stats["conversion"] = {"state": "Success"}

            try:
                with target_stats_file_path.open(mode="w") as stats_file:
                    # The sourceStats are now augmented with new info, so we write that to the target stats
                    json.dump(source_stats, stats_file)
            except Exception:
                # something went wrong writing the stats file, best to remove it and crash.
                logger.error("Failed writing the stats, removing the file to avoid inconsistent state")
                if target_stats_file_path.exists():
                    target_stats_file_path.unlink()
                raise
        except Exception:
            logger.error("Something went wrong executing the query, removing the output file")
            if target_stats_file_path.exists():
                target_stats_file_path.unlink()
            raise


def protobuf_builder(entmap: EntityMapper, relmap: RelationMapper) -> Type[QueryBuilder[pb_Query]]:

    class ProtobufQueryBuilder(QueryBuilder[pb_Query]):

        def __init__(self, number_of_triples: int, number_of_qualifiers: int) -> None:
            """
            Create
            """
            super().__init__(number_of_triples, number_of_qualifiers)
            self.query = pb_Query()
            for _ in range(number_of_triples):
                self.query.triples.append(pb_Triple())
            self._is_triple_set = [[False, False, False] for _ in range(number_of_triples)]
            for _ in range(number_of_qualifiers):
                self.query.qualifiers.append(pb_Qualifier())
            self._is_qual_set = [[False, False] for _ in range(number_of_qualifiers)]
            self.query.diameter = 0

        def set_subject(self, index: int, entity: str) -> None:
            assert not self._is_triple_set[index][0], f"The subject for triple with index {index} has already been set"
            entity_index = entmap.lookup(entity)
            # set normal edge
            self.query.triples[index].subject = entity_index
            self._is_triple_set[index][0] = True

        def set_entity_object(self, index: int, entity: str) -> None:
            assert not self._is_triple_set[index][2], f"The object for triple with index {index} has already been set"
            entity_index = entmap.lookup(entity)
            # we can be sure that the oneof has not been set because of the assert at the top of this method
            self.query.triples[index].object.entity = entity_index
            self._is_triple_set[index][2] = True

        def set_literal_object(self, index: int, literal_value: str) -> None:
            assert not self._is_triple_set[index][2], f"The object for triple with index {index} has already been set"
            # we can be sure that the oneof has not been set because of the assert at the top of this method
            self.query.triples[index].object.literal = literal_value
            self._is_triple_set[index][2] = True

        def set_predicate(self, index: int, predicate: str) -> None:
            assert not self._is_triple_set[index][1], f"The predicate for triple with index {index} has already been set"
            # set normal edge
            predicate_index = relmap.lookup(predicate)
            self.query.triples[index].predicate = predicate_index
            self._is_triple_set[index][1] = True

        def set_qualifier_rel(self, tripleIndex: int, qualifier_index: int, predicate: str) -> None:
            assert not self._is_qual_set[qualifier_index][0], f"The relation for qualifier with index {qualifier_index} has already been set"
            # set forward
            predicateIndex = relmap.lookup(predicate)
            self.query.qualifiers[qualifier_index].qualifier_relation = predicateIndex
            self.query.qualifiers[qualifier_index].corresponding_triple = tripleIndex
            self._is_qual_set[qualifier_index][0] = True

        def set_qualifier_entity_val(self, tripleIndex: int, qualifier_index: int, value: str) -> None:
            assert not self._is_qual_set[qualifier_index][1], f"The value for qualifier with index {qualifier_index} has already been set"
            valueIndex = entmap.lookup(value)
            # we can be sure that the oneof has not been set because of the assert at the top of this method
            self.query.qualifiers[qualifier_index].qualifier_value.entity = valueIndex
            self._is_qual_set[qualifier_index][1] = True

        def set_qualifier_literal_val(self, tripleIndex: int, qualifier_index: int, value: str) -> None:
            assert not self._is_qual_set[qualifier_index][1], f"The value for qualifier with index {qualifier_index} has already been set"
            # we can be sure that the oneof has not been set because of the assert at the top of this method
            self.query.qualifiers[qualifier_index].qualifier_value.literal = value
            self._is_qual_set[qualifier_index][1] = True

        def _set_easy_targets(self, values: Iterable[pb_EntityOrLiteral]) -> None:
            assert len(self.query.easy_targets) == 0, \
                "the list of targets can only be set once. If it is needed to create this incrementally, this implementations can be changed to first collect and only create the tensor in the final build."
            # TODO: this cannot be checked here since the pb types are not hashable!
            # assert len(list(values)) == len(set(values)), f"Values must be a set, got {values}"
            self.query.easy_targets.extend(values)

        def set_easy_entity_targets(self, values: Iterable[str]) -> None:
            self._set_easy_targets([pb_EntityOrLiteral(entity=entmap.lookup(value)) for value in values])

        def set_easy_literal_targets(self, values: Iterable[str]) -> None:
            self._set_easy_targets([pb_EntityOrLiteral(literal=value) for value in values])

        def _set_hard_targets(self, values: Iterable[pb_EntityOrLiteral]) -> None:
            assert len(self.query.hard_targets) == 0, \
                "the list of targets can only be set once. If it is needed to create this incrementally, this implementations can be changed to first collect and only create the tensor in the final build."
            # TODO: this cannot be checked here since the pb types are not hashable!
            # assert len(list(values)) == len(set(values)), f"Values must be a set, got {values}"
            assert len(list(values)) > 0, "Hard targets cannot be empty"
            self.query.hard_targets.extend(values)

        def set_hard_entity_targets(self, values: Iterable[str]) -> None:
            self._set_hard_targets([pb_EntityOrLiteral(entity=entmap.lookup(value)) for value in values])

        def set_hard_literal_targets(self, values: Iterable[str]) -> None:
            self._set_hard_targets([pb_EntityOrLiteral(literal=value) for value in values])

        def set_diameter(self, diameter: int) -> None:
            assert self.query.diameter == 0, "Setting the diameter twice is likely wrong"
            self.query.diameter = diameter

        def build(self) -> pb_Query:
            # checking that everything is filled
            for parts in self._is_triple_set:
                assert all(parts)
            for qvqr in self._is_qual_set:
                assert all(qvqr)
            assert self.query.diameter != 0
            return self.query

        @ staticmethod
        def get_file_extension() -> str:
            return ".proto"

        def store(self, collection: Iterable[pb_Query], absolute_target_path: Path) -> None:
            query_data = pb_QueryData()
            query_data.queries.extend(collection)
            with open(absolute_target_path, "wb") as output_file:
                output_file.write(query_data.SerializeToString())
    return ProtobufQueryBuilder


def torch_query_builder(entmap: EntityMapper, relmap: RelationMapper) -> Type[QueryBuilder[TorchQuery]]:

    class TorchQueryBuilder(QueryBuilder[TorchQuery]):
        """A builder for binary forms."""

        # This is a singleton used each time there are no qualifiers in the query
        __EMPTY_QUALIFIERS = torch.full((3, 0), -1, dtype=torch.long)

        targets: Optional[LongTensor]

        def __init__(self, number_of_triples: int, number_of_qualifiers: int) -> None:
            """
            Initialize the builder.
            """
            super().__init__(number_of_triples, number_of_qualifiers)
            self.number_of_triples = number_of_triples
            self.number_of_qualifiers = number_of_qualifiers
            # We initialize everything to -1. After adding the data there must not be a single -1 left.
            # Store the subject and object,
            self.edge_index = torch.full((2, number_of_triples), -1, dtype=torch.long)
            # Store the type of each edge,
            self.edge_type = torch.full((number_of_triples,), -1, dtype=torch.long)
            # Store all qualifiers. The first row has the qualifier relation, the second one the value and the third one the triple to which it belongs.
            if number_of_qualifiers == 0:
                self.qualifiers = TorchQueryBuilder.__EMPTY_QUALIFIERS
            else:
                self.qualifiers = torch.full((3, number_of_qualifiers), -1, dtype=torch.long)
            # the diameter of the query
            self.diameter = -1
            # The targets of the query. This size is unknown upfront, so we will just create it when set
            self.easy_targets: Optional[torch.Tensor] = None
            self.hard_targets: Optional[torch.Tensor] = None

        def set_subject(self, index: int, entity: str) -> None:
            entity_index = entmap.lookup(entity)
            # set normal edge
            self.set_subject_ID(index, entity_index)

        def set_subject_ID(self, index: int, entity_index: int) -> None:
            assert self.edge_index[0, index] == -1
            self.edge_index[0, index] = entity_index

        def set_entity_object(self, index: int, entity: str) -> None:
            entity_index = entmap.lookup(entity)
            # set normal edge
            self.set_entity_object_ID(index, entity_index)

        def set_literal_object(self, index: int, entity: str) -> None:
            raise NotImplementedError("Tensor builder does not support literals, use protocol buffers.")

        def set_entity_object_ID(self, index: int, entity_index: int) -> None:
            assert self.edge_index[1, index] == -1
            self.edge_index[1, index] = entity_index

        def set_predicate(self, index: int, predicate: str) -> None:
            # set normal edge
            predicate_index = relmap.lookup(predicate)
            self.set_predicate_ID(index, predicate_index)

        def set_predicate_ID(self, index: int, predicate_index: int) -> None:
            assert self.edge_type[index] == -1
            self.edge_type[index] = predicate_index

        def set_subject_predicate_entitiy_object_ID(self, triple_index: int, subject: int, predicate: int, object: int) -> None:
            self.set_subject_ID(triple_index, subject)
            self.set_predicate_ID(triple_index, predicate)
            self.set_entity_object_ID(triple_index, object)

        def set_qualifier_rel(self, triple_index: int, qualifier_index: int, predicate: str) -> None:
            # set forward
            predicate_index = relmap.lookup(predicate)
            self.qualifiers[0, qualifier_index] = predicate_index
            self.qualifiers[2, qualifier_index] = triple_index

        def set_qualifier_rel_ID(self, qualifier_index: int, predicateIndex: int, tripleIndex: int) -> None:
            self.qualifiers[0, qualifier_index] = predicateIndex
            self.qualifiers[2, qualifier_index] = tripleIndex

        def set_qualifier_entity_val(self, triple_index: int, qualifier_index: int, value: str) -> None:
            value_index = entmap.lookup(value)
            # set forward
            self.qualifiers[1, qualifier_index] = value_index

        def set_qualifier_literal_val(self, tripleIndex: int, index: int, value: str) -> None:
            raise NotImplementedError("TensorBuilder does not support literal values, use protocol buffers.")

        def set_qualifier_val_ID(self, qualifier_index: int, value_index: int, tripleIndex: int) -> None:
            self.qualifiers[1, qualifier_index] = value_index
            # note: tripleIndex is not set. We assume it is set in set_qualifier_rel_ID, eithe before or after this call.

        def OLDset_targets_ID(self, mapped: Iterable[int]) -> None:
            assert len(list(mapped)) == len(set(mapped))
            self.targets = torch.as_tensor(data=mapped, dtype=torch.long)

        def set_easy_entity_targets(self, values: Iterable[str]) -> None:
            assert self.easy_targets is None, "the list of targets can only be set once. If it is needed to create this incrementally, this implementations can be changed to first collect and only create the tensor in the final build."
            assert len(list(values)) == len(set(values)), f"Values must be a set got {values}"
            mapped = [entmap.lookup(value) for value in values]
            self.set_easy_entity_targets_ID(mapped)

        def set_easy_entity_targets_ID(self, mapped: Iterable[int]) -> None:
            assert len(list(mapped)) == len(set(mapped))
            self.easy_targets = torch.as_tensor(data=mapped, dtype=torch.long)

        def set_easy_literal_targets(self, values: Iterable[str]) -> None:
            raise NotImplementedError("TensorBuilder does not support literal values, use protocol buffers.")

        def set_hard_entity_targets(self, values: Iterable[str]) -> None:
            assert self.hard_targets is None, "the list of targets can only be set once. If it is needed to create this incrementally, this implementations can be changed to first collect and only create the tensor in the final build."
            assert len(list(values)) == len(set(values)), f"Values must be a set got {values}"
            mapped = [entmap.lookup(value) for value in values]
            self.set_hard_entity_targets_ID(mapped)

        def set_hard_entity_targets_ID(self, mapped: Iterable[int]) -> None:
            assert len(list(mapped)) == len(set(mapped))
            self.hard_targets = torch.as_tensor(data=mapped, dtype=torch.long)

        def set_hard_literal_targets(self, values: Iterable[str]) -> None:
            raise NotImplementedError("TensorBuilder does not support literal values, use protocol buffers.")

        def set_diameter(self, diameter: int) -> None:
            assert self.diameter == -1, "Setting the diameter twice is likely wrong"
            assert diameter <= self.number_of_triples, "the diameter of the query can never be larger than the number of triples"
            self.diameter = diameter

        def build(self) -> TorchQuery:
            # checkign that everything is filled
            assert (self.edge_index != -1).all()
            assert (self.edge_type != -1).all()
            assert (self.qualifiers != -1).all()
            assert self.diameter != -1
            assert self.easy_targets is not None
            assert self.hard_targets is not None

            return TorchQuery(self.edge_index, self.edge_type, self.qualifiers, self.easy_targets, self.hard_targets, torch.as_tensor(self.diameter))

        @ staticmethod
        def get_file_extension() -> str:
            return ".pickle"

        def store(self, collection: Iterable[TorchQuery], absolute_target_path: Path) -> None:
            torch.save(collection, absolute_target_path, pickle_module=dill, pickle_protocol=dill.HIGHEST_PROTOCOL)
    return TorchQueryBuilder


# @dataclasses.dataclass()
# class Triple:
#     """Representation of one non-qualified triple"""

#     # The subject of the triple
#     subject: str

#     # The predicate of the triple
#     predicate: str

#     # The object of the triple
#     object: str


# def StrippedSPARQLResultBuilder(sparql_endpoint=None, sparql_endpoint_options=None) -> None:
#     sparql_endpoint = sparql_endpoint or default_sparql_endpoint_address
#     sparql_endpoint_options = sparql_endpoint_options or default_sparql_endpoint_options

#     class StrippedSPARQLResultBuilderClass(BinaryFormBuilder[str]):  # type: ignore
#         def __init__(self, number_of_triples, number_of_qualifiers) -> None:

#             self.triples = [Triple(None, None, None) for _ in range(number_of_triples)]

#         def set_subject(self, index: int, entity: str) -> None:
#             self.triples[index].subject = entity

#         def set_object(self, index: int, entity: str) -> None:
#             self.triples[index].object = entity

#         def set_predicate(self, index: int, predicate: str) -> None:
#             self.triples[index].predicate = predicate

#         def set_qualifier_rel(self, tripleIndex: int, index: int, predicate: str) -> None:
#             """These get intentionally ignored in this converter"""
#             pass

#         def set_qualifier_val(self, tripleIndex: int, index: int, value: str) -> None:
#             """These get intentionally ignored in this converter"""
#             pass

#         def set_targets(self, values: Iterable[str]) -> None:
#             """These get intentionally ignored in this converter"""
#             pass

#         def set_diameter(self, diameter: int) -> None:
#             """These get intentionally ignored in this converter"""
#             pass

#         @staticmethod
#         def convertPossibleTarget(value: str) -> None:
#             if value == get_entity_mapper().get_target_entity_name():
#                 return "?target"
#             return value

#         @staticmethod
#         def convertPossibleEntity(value: str) -> None:
#             if value.startswith("?"):
#                 return value
#             return f"<{value}>"

#         def build(self) -> str:
#             start = "SELECT distinct ?target WHERE { graph <split:all> {"
#             bgp = ""
#             for triple in self.triples:
#                 bgp += f"{self.convertPossibleEntity(self.convertPossibleTarget(triple.subject))} "
#                 bgp += f"{self.convertPossibleEntity(self.convertPossibleTarget(triple.predicate))} "
#                 bgp += f"{self.convertPossibleEntity(self.convertPossibleTarget(triple.object))} . "
#             end = "} }"
#             query_string = start + bgp + end
#             return query_string

#         @staticmethod
#         def get_file_extension() -> str:
#             """Get the extension to be used for files stored with the store method. Each deriving class must implement its own static method."""
#             return ".txt"

#         def store(self, collection: Iterable[pb_Query], absolute_target_path: Path) -> None:  # type: ignore
#             triple_store = SPARQLStore(sparql_endpoint, returnFormat="csv", method="POST", **sparql_endpoint_options)  # headers={}
#             global_logger = logging.getLogger()
#             original_level = global_logger.getEffectiveLevel()
#             global_logger.setLevel(logging.ERROR)
#             try:
#                 with open(absolute_target_path, "w") as output_file:
#                     for query_string in collection:
#                         result = triple_store.query(query_string)
#                         separator = ""
#                         target_string = ""
#                         for target in list(result):
#                             target_string += (separator + target['target'])
#                             separator = "|"
#                         output_file.write(target_string + "\n")
#             finally:
#                 global_logger.setLevel(original_level)
#     return StrippedSPARQLResultBuilderClass
