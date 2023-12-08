"""
This module implements the loading of queries into a PyTorch Dataset.
The user can select the parts of the data to be loaded in test, training, and validation.
Besides, it allows the user to decide how many elements to take from each part of these sets.
"""
import dataclasses
import itertools
import json
import logging
from collections import defaultdict
from pathlib import Path
import pathlib
from typing import Any, Callable, DefaultDict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .conversion import torch_query_builder
# from .generated.query_pb2 import QueryData as pb_QueryData
from .types import LongTensor
from .dataset import Dataset as QuerySamplerDataSet
from .query_representation.torch import TorchQuery
from .query_representation.query_pb2 import QueryData as pb_QueryData
from .sample import Sample

__all__ = [
    "get_query_data_loaders",
    "Information",
    "DatafileInfo"
    # "QueryGraphBatch",
]

logger = logging.getLogger(__name__)


@dataclasses.dataclass()
class DatafileInfo:
    """Stores information about one specific file from which data is loaded"""
    source_file: Path
    amount_available: int
    amount_requested: int
    hashcode: str


class Information:
    def __init__(self) -> None:
        self.info: DefaultDict[str, List[Any]] = defaultdict(list)

    def append_info(self, split_name: str, data_root: pathlib.Path, sample: Sample, loaded: Iterable[DatafileInfo]) -> None:
        self.info[split_name].append(
            dict(
                selector=sample.selector,
                reify=sample.reify,
                remove_qualifiers=sample.remove_qualifiers,
                loaded=[
                    dict(
                        file=file_and_info.source_file.relative_to(data_root).as_posix(),
                        amount=file_and_info.amount_requested,
                        hash=file_and_info.hashcode,
                    )
                    for file_and_info in loaded
                ],
            )
        )

    def __str__(self) -> str:
        return self.info.__str__()


def get_query_datasets(
    # data_root: Path,
    dataset: QuerySamplerDataSet,
    train: Iterable[Sample],
    validation: Iterable[Sample],
    test: Iterable[Sample],
) -> Tuple[Mapping[str, Dataset[TorchQuery]], Information]:
    """
    This code is used to load datasets for training, validating and testing a model.

    For each of train, test, and validation, a list of Sample is expected,


    Note, the implementations does not check for overlap.
    Be careful that you do not accidentally load the same set twice.

    If the selector selects multiple datasets, the sampling will be uniform,
    taking into account the size of the datasets.
    If you want to select all items in a dataset, specify QuerySetLoader.ALL
    """
    data_root = dataset.query_proto_location()  # Path(data_root).expanduser().resolve()
    logger.info(f"Using data_root={data_root.as_uri()}")

    # the procedure is the same for each of the splits. Only the splits selected are different.
    datasets = dict()
    # store information about the actually loaded data.
    information = Information()
    for split_name, split in dict(
        train=train,
        validation=validation,
        test=test,
    ).items():
        # collect all samples
        all_samples = []
        for sample in split:
            total_available = 0
            datafiles_and_info: List[DatafileInfo] = []
            glob_pattern = "./" + sample.selector + "/**/*" + split_name + "_stats.json"
            stat_files = list(data_root.glob(glob_pattern))
            assert len(stat_files) > 0, f"The number of files for split {split_name} and pattern {sample.selector} was empty. This is close to always a mistake in the selector."
            for stats_file_path in stat_files:
                # for sample_path in data_root.glob("./" + sample.selector + "/*/"):
                #    for stats_file_path in sample_path.glob("*" + splitname + "_stats.json"):
                with open(stats_file_path) as stats_file:
                    stats = json.load(stats_file)
                amount_in_file = stats["count"]
                total_available += amount_in_file
                setname = stats["name"]
                # the setname already contains the splitname!
                data_file = stats_file_path.parents[0] / (setname + ".proto")
                assert data_file.exists(), \
                    f"The datafile {data_file} refered from {stats_file_path} could not be found."
                datafiles_and_info.append(DatafileInfo(data_file, amount_in_file, 0, stats["hash"]))

            # get how many we need, and then split proportionally
            requested_amount = sample.amount(total_available)
            fraction = requested_amount / total_available

            still_needed = requested_amount
            for datafile_and_info in datafiles_and_info:
                datafile_and_info.amount_requested = round(fraction * datafile_and_info.amount_available)
                still_needed -= datafile_and_info.amount_requested
            # correcting for possible rounding mistakes.
            for datafile_and_info in datafiles_and_info:
                if still_needed == 0:
                    break
                if still_needed > 0 and datafile_and_info.amount_available > datafile_and_info.amount_requested:
                    available_and_needed = min(datafile_and_info.amount_available - datafile_and_info.amount_requested, still_needed)
                    datafile_and_info.amount_requested += available_and_needed
                    still_needed -= available_and_needed
                if still_needed < 0 and datafile_and_info.amount_requested > 0:
                    to_remove = min(datafile_and_info.amount_requested, -still_needed)
                    datafile_and_info.amount_requested -= to_remove
                    still_needed += to_remove
                # datafiles_and_info[-1].amount_requested += still_needed
            assert still_needed == 0

            information.append_info(split_name, data_root, sample, datafiles_and_info)

            for datafile_and_info in datafiles_and_info:
                all_samples.append(__OneFileDataset(dataset, datafile_and_info.source_file, datafile_and_info.amount_requested, sample.reify, sample.remove_qualifiers))
        if all_samples:
            concatenated_dataset = torch.utils.data.ConcatDataset(all_samples) if split else __EmptyDataSet()
            datasets[split_name] = concatenated_dataset
        else:
            logger.error(f"No samples for split {split_name}")
    return datasets, information


def read_queries_from_proto(
    dataset: QuerySamplerDataSet,
    input_path: Path,
    reify: bool,
    remove_qualifiers: bool,
) -> Iterable[TorchQuery]:
    """Yield query data from a protobuf."""
    assert not (reify and remove_qualifiers), "cannot both reify and remove qualifiers"
    if reify:
        yield from read_queries_from_proto_with_reification(input_path, dataset)
    else:
        yield from read_queries_from_proto_without_reification(input_path, remove_qualifiers)


def read_queries_from_proto_with_reification(
    input_path: Path,
    dataset: QuerySamplerDataSet
) -> Iterable[TorchQuery]:
    """
    Read preprocessed queries from Protobuf file.
    Then, all triples are reified and qualifiers are added as properties of the blank nodes.

    :param input_path:
        The input file path.

    :yields:
        A query data object.
    """
    relmap, entmap = dataset.get_mappers()
    query_builder = torch_query_builder(relmap, entmap)
    input_path = Path(input_path).expanduser().resolve()
    logger.info(f"Reading from {input_path}")

    qd = pb_QueryData()
    with input_path.open("rb") as input_file:
        qd.ParseFromString(input_file.read())

    logger.error("Reification does currently just take the diameter from the source query without modification")

    for query in qd.queries:
        # we reify by reifying each triple and then adding a triple for each qualifier.
        number_of_original_triples = len(query.triples)
        number_of_triples = number_of_original_triples * 3 + len(query.qualifiers)
        number_of_qualifiers = 0
        b = query_builder(number_of_triples, number_of_qualifiers)

        # Reified statement need 3 triples, so we need to track where we are in the builder. we do not have qualifiers, so we do not need to track where the triples are
        for (index, triple) in enumerate(query.triples):
            statement_entity_index = entmap.get_reified_statement_index(index)
            builder_triple_index = 3 * index

            b.set_subject_predicate_entitiy_object_ID(builder_triple_index, statement_entity_index, relmap.reified_subject_index, triple.subject)

            predicate_entity = entmap.get_entity_for_predicate(triple.predicate)
            b.set_subject_predicate_entitiy_object_ID(builder_triple_index + 1, statement_entity_index, relmap.reified_predicate_index, predicate_entity)

            assert triple.object.HasField("entity"), "Converting queries with literal values to tensors is not supported."
            b.set_subject_predicate_entitiy_object_ID(builder_triple_index + 2, statement_entity_index, relmap.reified_object_index, triple.object.entity)

        # build qualifiers. These are now just added as triples
        for (builder_triple_index, qualifier) in zip(range(number_of_original_triples * 3, number_of_triples), query.qualifiers):
            statement_entity_index = entmap.get_reified_statement_index(qualifier.corresponding_triple)
            assert qualifier.qualifier_value.HasField("entity")
            b.set_subject_predicate_entitiy_object_ID(builder_triple_index, statement_entity_index, qualifier.qualifier_relation, qualifier.qualifier_value.entity)

        # set diameter and targets
        b.set_diameter(query.diameter)
        for targets, assign_to in [(query.easy_targets, b.set_easy_entity_targets_ID), (query.hard_targets, b.set_hard_entity_targets_ID)]:
            entity_targets: List[int] = []
            for t in targets:
                assert t.HasField("entity"), "Converting queries with literal values to tensors is not supported."
                entity_targets.append(t.entity)
            assign_to(entity_targets)

        # build the object
        yield b.build()


def read_queries_from_proto_without_reification(
    input_path: Path,
    remove_qualifiers: bool,
) -> Iterable[TorchQuery]:
    """
    Read preprocessed queries from Protobuf file.

    :param input_path:
        The input file path.
    :param remove_qualifiers:
        Whether to remove qualifiers.

    :yields:
        A query data object.
    """

    query_builder = torch_query_builder(None, None)
    input_path = Path(input_path).expanduser().resolve()
    logger.info(f"Reading from {input_path}")

    qd = pb_QueryData()
    with input_path.open("rb") as input_file:
        qd.ParseFromString(input_file.read())

    for query in qd.queries:
        number_of_triples = len(query.triples)
        if remove_qualifiers:
            number_of_qualifiers = 0
        else:
            number_of_qualifiers = len(query.qualifiers)
        b = query_builder(number_of_triples, number_of_qualifiers)

        # build triples
        for (index, triple) in enumerate(query.triples):
            assert triple.object.HasField("entity"), "Converting queries with literal values to tensors is not supported."
            b.set_subject_predicate_entitiy_object_ID(index, triple.subject, triple.predicate, triple.object.entity)
        if not remove_qualifiers:
            # build qualifier
            for (index, qualifier) in enumerate(query.qualifiers):
                b.set_qualifier_rel_ID(index, qualifier.qualifier_relation, qualifier.corresponding_triple)
                assert qualifier.qualifier_value.HasField("entity"), "Converting queries with literal values to tensors is not supported."
                b.set_qualifier_entity_val_ID(index, qualifier.qualifier_value.entity, qualifier.corresponding_triple)

        # set diameter and targets
        b.set_diameter(query.diameter)
        for targets, assign_to in [(query.easy_targets, b.set_easy_entity_targets_ID), (query.hard_targets, b.set_hard_entity_targets_ID)]:
            entity_targets: List[int] = []
            for t in targets:
                assert t.HasField("entity"), "Converting queries with literal values to tensors is not supported."
                entity_targets.append(t.entity)
            assign_to(entity_targets)

        # build the object
        yield b.build()


class __EmptyDataSet(Dataset[TorchQuery]):
    def __getitem__(self, index: int) -> TorchQuery:
        raise KeyError(index)

    def __len__(self) -> int:
        return 0


class __OneFileDataset(Dataset[TorchQuery]):
    """A query dataset from one file (=one query pattern)."""

    def __init__(self, dataset: QuerySamplerDataSet, path: Path, amount: int, reify: bool, remove_qualifiers: bool) -> None:
        assert not (reify and remove_qualifiers), "Cannot both remove qualifiers and reify them."
        super().__init__()
        self.data = list(itertools.islice(read_queries_from_proto(dataset, path, reify, remove_qualifiers), amount))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> TorchQuery:
        return self.data[item]


def _unique_with_inverse(*tensors: LongTensor) -> Tuple[LongTensor, Sequence[LongTensor]]:
    # get unique IDs over all these tensors, and the flat inverse
    concatenated: torch.Tensor = torch.cat([t.view(-1) for t in tensors], dim=0)
    unique, flat_inverse = concatenated.unique(return_inverse=True)  # type: ignore

    # decompose inverse into the individual tensors
    inverse_flat_tensors = flat_inverse.split([t.numel() for t in tensors])
    inverse_list = [
        it.view(*t.shape)
        for t, it in zip(
            tensors,
            inverse_flat_tensors,
        )
    ]

    return unique, inverse_list


@ dataclasses.dataclass
class QueryGraphBatch:
    """A batch of query graphs."""

    #: The global entity IDs occurring in this batch. Their order corresponds to the batch-local entity ID,
    #: i.e. `local_entity_id = i` corresponds to global entity ID `global_entity_ids[i]`.
    #: shape: (num_unique_batch_entities,)
    entity_ids: LongTensor

    #: The global relation IDs occurring in this batch. Their order corresponds to the batch-local relation ID.
    #: shape: (num_unique_batch_relations,)
    relation_ids: LongTensor

    #: The edge index of the batch graph (in batch-local entity IDs)
    #: shape: (2, num_batch_edges)
    edge_index: LongTensor

    #: The edge types of the batch graph (in batch-local relation IDs)
    #: shape: (num_batch_edges,)
    edge_type: LongTensor

    #: The qualifier index of the batch graph (in batch-local relation/entity/edge IDs)
    #: shape: (3, num_batch_qualifier_pairs)
    qualifier_index: LongTensor

    #: shape: (num_unique_batch_entities,)
    #: The index which maps nodes to a particular query/graph, e.g., [0,0,0,1,1,1] for 6 nodes of two queries
    graph_ids: LongTensor

    #: shape: (batch_size,)
    #: The batched query diameters.
    query_diameter: LongTensor

    #: The easy targets, in format of pairs (graph_id, entity_id)
    #: shape: (2, num_targets)
    easy_targets: LongTensor

    #: The hard targets, in format of pairs (graph_id, entity_id)
    #: shape: (2, num_targets)
    hard_targets: LongTensor

    def __post_init__(self) -> None:
        assert self.entity_ids is not None
        assert self.relation_ids is not None
        assert self.edge_type is not None
        assert self.qualifier_index is not None
        assert self.graph_ids is not None
        assert self.query_diameter is not None
        assert self.easy_targets is not None
        assert self.hard_targets is not None


def collate_query_data(dataset: QuerySamplerDataSet) -> Callable[[Sequence[TorchQuery]], QueryGraphBatch]:
    def _collate_query_data(
        batch: Sequence[TorchQuery],
    ) -> QueryGraphBatch:
        """
        A collator for query graph batches.

        :param batch:
            The sequence of batch elements.

        :return:
            A QueryGraphBatch with all colated data
        """
        _, entmap = dataset.get_mappers()
        global_entity_ids = []
        global_relation_ids = []
        edge_index = []
        edge_type = []
        qualifier_index = []
        query_diameter = []
        easy_targets: List[LongTensor] = []
        hard_targets: List[LongTensor] = []

        entity_offset = relation_offset = edge_offset = 0
        for i, query_data in enumerate(batch):
            # append query diameter
            query_diameter.append(query_data.query_diameter)

            # convert to local ids
            global_entity_ids_, (local_edge_index, local_qualifier_entities) = _unique_with_inverse(
                query_data.edge_index,
                query_data.qualifier_index[1],
            )

            # target nodes
            target_mask = global_entity_ids_ == entmap.get_target_index()
            global_entity_ids_[target_mask] = entmap.number_of_entities_and_reified_relations_without_vars_and_targets() + 1

            # variable nodes
            var_mask = (global_entity_ids_ >= entmap.variable_start) & (global_entity_ids_ < entmap.variable_end)
            global_entity_ids_[var_mask] = entmap.number_of_entities_and_reified_relations_without_vars_and_targets() + 2

            # blank nodes
            reification_mask = global_entity_ids_ >= entmap.reified_statements_start
            global_entity_ids_[reification_mask] = entmap.number_of_entities_and_reified_relations_without_vars_and_targets() + 3

            global_relation_ids_, (local_edge_type, local_qualifier_relations) = _unique_with_inverse(
                query_data.edge_type,
                query_data.qualifier_index[0],
            )

            # add offsets: entities ...
            local_edge_index = local_edge_index + entity_offset  # type: ignore[assignment]
            local_qualifier_entities = local_qualifier_entities + entity_offset  # type: ignore[assignment]
            # ... relations ...
            local_edge_type = local_edge_type + relation_offset  # type: ignore[assignment]
            local_qualifier_relations = local_qualifier_relations + relation_offset  # type: ignore[assignment]
            # ... and edge ids
            batch_edge_ids = query_data.qualifier_index[2] + edge_offset

            # re-compose qualifier index
            local_qualifier_index = torch.stack(
                [
                    local_qualifier_relations,
                    local_qualifier_entities,
                    batch_edge_ids,
                ],
                dim=0,
            )

            # append
            global_entity_ids.append(global_entity_ids_)
            global_relation_ids.append(global_relation_ids_)
            edge_index.append(local_edge_index)
            edge_type.append(local_edge_type)
            qualifier_index.append(local_qualifier_index)

            # increase counters
            entity_offset += len(global_entity_ids_)
            relation_offset += len(global_relation_ids_)
            edge_offset += len(local_edge_type)

            # collate easy targets
            easy_targets.append(torch.stack([
                torch.full_like(query_data.easy_targets, fill_value=i),
                query_data.easy_targets,
            ]))

            # collate hard targets
            hard_targets.append(torch.stack([
                torch.full_like(query_data.hard_targets, fill_value=i),
                query_data.hard_targets,
            ]))

        # concatenate
        global_entity_ids_t = torch.cat(global_entity_ids, dim=-1)
        global_relation_ids_t = torch.cat(global_relation_ids, dim=-1)
        edge_index_t = torch.cat(edge_index, dim=-1)
        edge_type_t = torch.cat(edge_type, dim=-1)
        qualifier_index_t = torch.cat(qualifier_index, dim=-1)
        query_diameter_t = torch.as_tensor(query_diameter, dtype=torch.long)
        easy_targets_t = torch.cat(easy_targets, dim=-1)
        hard_targets_t = torch.cat(hard_targets, dim=-1)

        # add graph ids
        graph_ids = torch.empty_like(global_entity_ids_t)
        start = 0
        for i, ids in enumerate(global_entity_ids):
            stop = start + len(ids)
            graph_ids[start:stop] = i
            start = stop

        return QueryGraphBatch(
            entity_ids=global_entity_ids_t,
            relation_ids=global_relation_ids_t,
            edge_index=edge_index_t,
            edge_type=edge_type_t,
            qualifier_index=qualifier_index_t,
            graph_ids=graph_ids,
            query_diameter=query_diameter_t,
            easy_targets=easy_targets_t,
            hard_targets=hard_targets_t,
        )
    return _collate_query_data


def get_query_data_loaders(
    dataset: QuerySamplerDataSet,
    train: Iterable[Sample],
    validation: Iterable[Sample],
    test: Iterable[Sample],
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[Mapping[str, torch.utils.data.DataLoader[TorchQuery]], Information]:
    """
    Get data loaders for query datasets.

    :param batch_size:
        The batch size to use for all data loaders.
    :param num_workers:
        The number of worker processes to use for loading. 0 means that the data is loaded in the main process.
    :param kwargs:
        Additional keyword-based arguments passed to ``get_query_datasets``.

    :return:
        A pair loaders, information, where loaders is a dictionary from split names to the data loaders, and information
        is a dictionary comprising information about the actually loaded data.
    """
    data_splits, information = get_query_datasets(dataset, train, validation, test)
    loaders: Mapping[str, DataLoader[TorchQuery]] = {
        splitname: DataLoader(
            dataset=data_split,
            batch_size=batch_size,
            shuffle=splitname == "train",
            collate_fn=collate_query_data(dataset),
            pin_memory=True,
            drop_last=splitname == "train",
            num_workers=num_workers,
        )
        for splitname, data_split in data_splits.items()
    }
    return loaders, information
