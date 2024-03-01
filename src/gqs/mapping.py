"""Mapping between RDF identifiers (IRIs) and numeric indices used in tensors"""


import logging
import pathlib
from typing import Any, Dict, Iterable, Tuple
import gqs.dataset
from rdflib.query import ResultRow

from ._sparql_execution import execute_sparql_to_result_silenced

# from tqdm import tqdm

__all__ = [
    "create_mapping",
    "get_entity_mapper",
    "get_relation_mapper",
    "EntityMapper",
    "RelationMapper",
]

logger = logging.getLogger(__name__)


class RelationMapper:
    """
    This RelationMapper manages the identifiers of relations and relation variables.
    It is initialized with an ordered list of all used relation identifiers and then deterministicallly provides unique numeric indices for these.
    Besides, it provides indices for inverse predicates as well as special predicates for relation variables and those used to reify stataments.
    """

    def __init__(self, predicates: Iterable[str]) -> None:
        """Initialize the mapper."""
        # First space for the normal predicates
        self._predicate_start = 0
        self._normal_predicate_count = len(list(predicates))
        assert self._normal_predicate_count > 0, "Trying to create a Relationmapper with 0 predicates might is most likely an error."
        assert self._normal_predicate_count == len(set(predicates)), "The identifiers in the predicates list were not unique"
        self._mapping = {val: i for (i, val) in enumerate(predicates)}
        self._normal_predicate_end = self._normal_predicate_count - 1
        # Then the 3 reification predicates
        self.reified_subject_index = self._normal_predicate_end + 1
        self.reified_predicate_index = self._normal_predicate_end + 2
        self.reified_object_index = self._normal_predicate_end + 3
        # Then we have all the inverses
        self._inverse_start = self._normal_predicate_end + 4
        # Note: also the refied ones have an inverse.
        self._inverse_end = self._inverse_start + self._normal_predicate_count + 3 - 1

        self._variable_start = self._inverse_end + 1

    def __str__(self) -> str:
        return f"""Relation mapper with {self._normal_predicate_count} predicates [0, {self._normal_predicate_end}],
        reified subject index : {self.reified_subject_index},
        reified predicate index: {self.reified_predicate_index},
        reified object index : {self.reified_object_index},
        {self._normal_predicate_count} normal inverse predicates and 3 reified inverses [{self._inverse_start}, {self._inverse_end}],
        and infinite variables [{self._variable_start}, ...]
        """

    def number_of_relation_types(self) -> int:
        """Get the number of (forward) relation types this relaiton mapper was initialized with. Note that this does nto include inverse relations, nor relations for reification and variables"""
        return self._normal_predicate_count

    def lookup(self, relation: str) -> int:
        """
        Get the numeric index of a relation.

        Warning: it is not possible to get a representation fo the reified statement IDs this way!

        arguments
        ----------
        relation: str
            The name of the relation
        """
        return self._mapping[relation]

    def inverse_lookup(self, relation_id: int) -> str:
        """Get the string representation of the relation with the given id. This is the inverse function of lookup.

        Warning: this can only be used to lookup the string form of forward relations.

        Args:
            relation_id (int): the id of the relation

        Returns:
            str: the str which was used as the identifier/label of this relation
        """
        assert relation_id > 0 and relation_id < self.get_largest_forward_relation_id(), "inverse_lookup can only be used for forward relations, not for inverses, etc"
        # created lazily as most applications won't ever call this
        try:
            mapping: dict[int, str] = self._inverse_mapping
        except AttributeError:
            self._inverse_mapping: dict[int, str] = {v: k for k, v in self._mapping.items() if v < self.get_largest_forward_relation_id()}
            mapping = self._inverse_mapping
        return mapping[relation_id]

    def get_inverse_of_index(self, relation_index: int) -> int:
        """
        Get the numeric index of a the inverse of the specified relation. The specified relation must be obtained from an earlier lookup call.
        Warning: this can only give the inverse of a forward relation. You cannot get the forward relation of one which has been inverted before.
        Warning: it is not possible to get a representation fo the reified statement IDs this way!

        arguments
        ----------
        relation_index: int
            The relation index.
        """
        assert relation_index >= 0 and relation_index < self._inverse_start
        return relation_index + self._inverse_start

    def get_inverse_relation(self, relation: str) -> int:
        """
        Get the numeric index of the inverse of a relation.
        Warning: it is not possible to get a representation for the inverse of reified statement IDs this way!

        arguments
        ----------
        relation: str
            The name of the relation
        """
        relation_index = self.lookup(relation)
        return self.get_inverse_of_index(relation_index)

    def get_largest_forward_relation_id(self) -> int:
        """Return the largest forward relation ID. Does not include inverse relations."""
        return self._normal_predicate_end + 3


class EntityMapper:
    """
    Object used to map entity and entity variables to indices.

    The entities are initialized determinisically with the ordered entities provided during construction.
    This mapper also maintains a mapping for entities representing the relations maintained by the `RelationMapper` this `EntityMapper` gets constructed with.
    This mapping also maintaind the index of the target variable and makes sure it does not collide with the other indices.
    The variables are given an index upon first use of a variable. Variable names must be of the form "varX" where X is an integer.
    """

    MAX_VARIABLE_COUNT = 10000
    """The maximum number of variables supported by this mapper. This is then also the maximum amount a query can have.
    This needs is limited because also statement nodes for reification get an identifier, which starts after the variables
    """

    def __init__(self, entities: Iterable[str], relation_mapper: RelationMapper) -> None:
        """
        This creates an EntityMapper. The ownership of the entityMapping moves to this object and will be modified.
        """
        # The normal entities are at the start
        self._normal_entity_start = 0
        self._normal_entity_count = len(list(entities))
        assert self._normal_entity_count == len(set(entities)), "The identifiers in the entities list were not unique"
        self._entitiy_and_var_mapping = {val: i for (i, val) in enumerate(entities)}
        self._normal_entity_end = self._normal_entity_count - 1

        # Then we have one entity for all real, forward relations
        # We keep the relation mapping separatly because there could be a collission with the identifiers in self._entity_mapping
        self._relation_entity_start = self._normal_entity_end + 1
        self._relation_mapping = {}
        for relation_index in relation_mapper._mapping.values():
            # we need to offset all relations by self._relation_entity_start
            index_with_offset = self._relation_entity_start + relation_index
            self._relation_mapping[relation_index] = index_with_offset
            # We also add entities for inverse relations, we look up the
            inverse_relation_index = relation_mapper.get_inverse_of_index(relation_index)
            inverse_relation_index_with_offset = inverse_relation_index + self._relation_entity_start
            self._relation_mapping[inverse_relation_index] = inverse_relation_index_with_offset

        self._relation_entity_end = self._relation_entity_start + 2 * relation_mapper._normal_predicate_count

        # all_relation_indices = relation_mapper.mapping.values()
        # number_of_relation_indices = len(all_relation_indices)
        # for index, relation_index in enumerate(all_relation_indices):
        #     self.relation_entities[relation_index] = index + self._highest_real_entity
        # # also add all inverse.
        # for index, relation_index in enumerate(all_relation_indices):
        #     inverse_reltion_index = relation_mapper.get_inverse_of_index(relation_index)
        #     self.relation_entities[inverse_reltion_index] = index + number_of_relation_indices + self._highest_real_entity

        # we do *not* add special relations for reification itself. This is intentional.
        # They would only occur by double reification, which might lead to some other issues.

        # self.highest_entity_index = self._highest_real_entity + 2 * number_of_relation_indices

        # Rounding up here. There is no particular reason for this, except that it is easier to recognize while debugging
        next_rounded = ((self._relation_entity_end // 10000) + 1) * 10000
        self.target_index = next_rounded
        self._entitiy_and_var_mapping[EntityMapper.get_target_entity_name()] = self.target_index
        # Also here, the only reaosn for the offset is for ease of debugging
        self.variable_start = self.target_index + 10000
        self.variable_end = self.variable_start + EntityMapper.MAX_VARIABLE_COUNT
        self.reified_statements_start = self.variable_end + 1

    def __str__(self) -> str:
        return f"""Entity mapper with {self._normal_entity_count} entities [0,{self._normal_entity_end}],
        Then, reified relations and their inverses [{self._relation_entity_start}, {self._relation_entity_end}]
        The target index is {self.target_index}
        Then, {EntityMapper.MAX_VARIABLE_COUNT} variables [{self.variable_start}, {self.variable_end}]
        Then infinite reified statements [{self.reified_statements_start}, ...]
        """

    def number_of_real_entities(self) -> int:
        """Get the number of entities this mapper was initialized with. This does not include entities representing relations, nor entities for variables and tartgets """
        return self._normal_entity_count

    def number_of_entities_and_reified_relations_without_vars_and_targets(self) -> int:
        """Get the number of entities this mapper was initialized with and the number of entities used for representing relations. This does not include entities for variables, tartgets, and reified statement."""
        return self._relation_entity_end

    def lookup(self, entity: str) -> int:
        """Lookup the ID of an entity."""
        index = self._entitiy_and_var_mapping.get(entity)
        if index is not None:
            return index
        # it is not in the index, so it must be a variable which has not been seen before
        assert entity.startswith("?var"), f"Got a entity that was not in the mapping and which is not a variable starting with \"var\". Key = {entity}"
        number = entity[4:]
        try:
            asInt = int(number)
            assert asInt >= 0, f"The index of the variable parsed as an integer, but the number was negative. Key = {entity}"
            assert asInt < EntityMapper.MAX_VARIABLE_COUNT, \
                f"The mapper only support up to {EntityMapper.MAX_VARIABLE_COUNT} variables, while the specified variable goes beyond that. Key = {entity}"
        except ValueError:
            raise Exception(f"Got a key which is not in the mapping, starts with var, but does not have a number after it. Key = {entity}")
        index = self.variable_start + asInt
        # We add this as a new mapping so it can be used directly the next time it is needed.
        self._entitiy_and_var_mapping[entity] = index
        return index

    def inverse_lookup(self, id: int) -> str:
        """Get the string representation of the entity with the given id. This is the inverse function of lookup.

        Warning: this can only be used to lookup the string form of real entities and not of variables nor reified things.

        Args:
            relation_id (int): the id of the entity

        Returns:
            str: the str which was used as the identifier/label of this entity
        """
        assert id > 0 and id and id < self.number_of_real_entities(), "inverse_lookup can only be used for real entities, not for variables"
        # created lazily as most applications won't ever call this
        try:
            mapping: dict[int, str] = self._inverse_entity_mapping
        except AttributeError:
            self._inverse_entity_mapping: dict[int, str] = {v: k for k, v in self._entitiy_and_var_mapping.items() if v < self.number_of_real_entities()}
            mapping = self._inverse_entity_mapping
        return mapping[id]

    def get_reified_statement_index(self, statement_index: int) -> int:
        """
        Get the entity index for a reified statement.

        This method guarantees that for the same inputindex, you always receive the same entity index.
        Besides, it guarantees that these entity indices do not collide with existing indexes for entities and entities used to represent relations.
        A final guarantee is that `is_entity_variable` will return True for these indices.

        These indices are only unique within the scope of a graph, e.g., a query.

        Arguments
        ---------
        statement_index: int
            The index of the statement. Typically this is the index of the statement within the query.

        Returns
        -------
        int
            The entity index for this statement
        """
        return self.reified_statements_start + statement_index

    def get_entity_for_predicate(self, predicate_index: int) -> int:
        """
        Get the entity representing the predicate with the index `predicate_index`.
        This predicate_index is to be obtained from the `RelationMapper` this `EntityMapper` was constructed with.
        Also inverted edges have an entity representation, so the predicate_index can be obtained from `RelationMapper.get_inverse_index_of`
        """
        return self._relation_mapping[predicate_index]

    def get_target_index(self) -> int:
        """Return the target index."""
        return self.target_index

    @staticmethod
    def get_target_entity_name() -> str:
        """Return the name used for the target entity."""
        return "TARGET"

    @staticmethod
    def is_valid_variable_name(entity: str) -> bool:
        """Check whether an entity name is a valid variable name."""
        if not entity.startswith("?var"):
            return False
        number = entity[4:]
        try:
            as_int = int(number)
            if as_int > EntityMapper.MAX_VARIABLE_COUNT:
                raise Exception(f"The entity mapper only supports up to {EntityMapper.MAX_VARIABLE_COUNT} variables")
            elif as_int >= 0:
                return True
        except ValueError:
            pass
        return False

    def is_entity_variable(self, entity_id: int) -> bool:
        """
        Check whether there is a variable at the given index. Also, returns True for reified statement entities.

        Note that this only checks whether that index could potentially have been used for a variable, it does not guarantee that it has been used before.
        """
        return entity_id >= self.variable_start

    def is_entity_reified_statement(self, entity_id: int) -> bool:
        """
        Returns True in case the id is an index corresponding to a refied statement.
        Note, that this cannot check whether that reified statement really exists, only that it could potentially have been used.
        """
        return entity_id >= self.reified_statements_start

    def is_entity_target(self, entity_id: int) -> bool:
        """Check whether an entity ID is the target entity."""
        return entity_id == self.target_index

    # This method is not really in accordance with the rest of the mapper because
    # it assumes one embedding for target, one for variable and one for blank node.
    # This is a design choice outside of this mapper
    # def get_largest_embedding_id(self) -> int:
    #     """Return the largest embedding ID (entities, target, variable, and reification blank node included).

    #     """
    #     return self._normal_entity_end + 3  # 3 = target/variable/reification blank node


def create_mapping(dataset: "gqs.dataset.Dataset", sparql_endpoint: str, sparql_endpoint_options: Dict[str, Any]) -> None:
    def query_entities_to_file(query: str, file: pathlib.Path, var: str) -> None:
        entities: list[str] = []
        for result in execute_sparql_to_result_silenced(query, sparql_endpoint, sparql_endpoint_options):
            assert isinstance(result, ResultRow)
            entities.append(str(result.get(var)))
        entities.sort()
        with open(file, "wt") as output_file:
            output_file.write('\n'.join(entities))
    if dataset.mapping_location().exists():
        raise Exception("The folder for mappings already exists, remove it first. Aborting.")
    dataset.mapping_location().mkdir(parents=True, exist_ok=False)
    entities_query = """
    select distinct ?entity where
    {
        graph<split:all>{
            {?entity ?_p ?_o} UNION {?_s ?_p ?entity} UNION { <<?_s ?_p ?_o>> ?_qr ?entity }
        }
        FILTER(isIRI(?entity))
    }"""
    query_entities_to_file(entities_query, dataset.entity_mapping_location(), "entity")

    relations_query = """
    select distinct ?relation where
    {
        graph<split:all>{
            {?_s ?relation ?_o} UNION { <<?_s ?_p ?_o>> ?relation ?_o }
        }
        FILTER(isIRI(?relation))
    }"""
    query_entities_to_file(relations_query, dataset.relation_mapping_location(), "relation")


def remove_mapping(dataset: "gqs.dataset.Dataset") -> None:
    dataset.relation_mapping_location().unlink(missing_ok=True)
    dataset.entity_mapping_location().unlink(missing_ok=True)
    if dataset.mapping_location().exists():
        dataset.mapping_location().rmdir()


def mapping_exists(dataset: "gqs.dataset.Dataset") -> bool:
    if not dataset.mapping_location().exists():
        return False
    if not (dataset.relation_mapping_location().is_file() and dataset.relation_mapping_location().exists()):
        return False
    if not (dataset.entity_mapping_location().is_file() and dataset.entity_mapping_location().exists()):
        return False
    return True

    # dataset.relation_mapping_location().unlink(missing_ok=True)
    # dataset.entity_mapping_location().unlink(missing_ok=True)
    # if dataset.mapping_location().exists():
    #     dataset.mapping_location().rmdir()


def get_mappers(dataset: "gqs.dataset.Dataset") -> Tuple[RelationMapper, EntityMapper]:
    relmap = get_relation_mapper(dataset)
    entmap = get_entity_mapper(dataset, relmap)
    return relmap, entmap


def get_relation_mapper(dataset: "gqs.dataset.Dataset") -> RelationMapper:
    if not dataset.relation_mapping_location().exists():
        raise Exception("Trying to create a relation mapper, but the mapping has not been created yet.")
    with open(dataset.relation_mapping_location()) as relation_file:
        relations = [relation.strip() for relation in relation_file.readlines()]
        return RelationMapper(relations)


def get_entity_mapper(dataset: "gqs.dataset.Dataset", relmap: RelationMapper) -> EntityMapper:
    if not dataset.entity_mapping_location().exists():
        raise Exception("Trying to create an entity mapper, but the mapping has not been created yet.")
    with open(dataset.entity_mapping_location()) as entity_file:
        entities = [entity.strip() for entity in entity_file.readlines()]
        return EntityMapper(entities, relmap)

# def _pad_statements_(data: List[list], max_len: int) -> List[list]:
#     """Padding index is always 0 as in the embedding layers of models. Cool? Cool. """
#     result = [
#         statement + [0] * (max_len - len(statement)) if len(statement) < max_len else statement[:max_len]
#         for statement in data
#     ]
#     return result


# def __load_mapping(from_file_name) -> Mapping[str, int]:
#     """Load mapping from a file."""
#     mapping = {}
#     with open(from_file_name) as fromfile:
#         for line in fromfile:
#             parts = line.split("\t")
#             try:
#                 index = int(parts[1])
#             except ValueError:
#                 raise Exception(f"In {from_file_name} there is an entry {parts[0]} with an associated index that is not an int. got {parts[1]}.")
#             mapping[parts[0]] = index
#     return mapping


# def __sort_map_pairs_by_numeric_string_key(the_map: Mapping[str, int]) -> List[Tuple[str, int]]:
#     """Sort mapping pairs by a numeric string key."""
#     as_pairs = list(the_map.items())

#     def get_key_as_number(pair) -> None:
#         if pair[0] == "__na__":
#             return 0
#         return int(pair[0][1:])

#     as_pairs.sort(key=get_key_as_number)
#     return as_pairs


# T = TypeVar("T", str, int)


# def _get_uniques_(
#     *data: Iterable[Sequence[T]],
# ) -> Tuple[List[T], List[T]]:
#     """ Throw in parsed_data/wd50k/ files and we'll count the entities and predicates"""
#     entities: Set[T] = set()
#     relations: Set[T] = set()
#     for statement in itertools.chain(*data):
#         entities.update(statement[0::2])
#         relations.update(statement[1::2])
#     return sorted(entities), sorted(relations)


# def remove_dups(data: List[list]) -> List[list]:
#     """
#     Remove duplicates.

#     :param data: list of lists with possible duplicates
#     :return: a list without duplicates
#     """
#     new_l = []
#     for datum in tqdm(data):
#         if datum not in new_l:
#             new_l.append(datum)

#     return new_l


# def load_clean_wd50k(name: str, subtype: str, max_len: int = 43) -> Dict:
#     """
#         :return: train/valid/test splits for the wd50k datasets
#     """
#     assert name in ['wd50k', 'wd50k_100', 'wd50k_33', 'wd50k_66'], \
#         "Incorrect dataset"
#     assert subtype in ["triples", "quints", "statements"], "Incorrect subtype: triples/quints/statements"
#     data_path = PurePath(old_data_root, "clean", name, subtype)

#     # Load raw data
#     with open(data_path / 'train.txt', 'r') as f:
#         raw_trn = []
#         for line in f.readlines():
#             raw_trn.append(line.strip("\n").split(","))

#     with open(data_path / 'test.txt', 'r') as f:
#         raw_tst = []
#         for line in f.readlines():
#             raw_tst.append(line.strip("\n").split(","))

#     with open(data_path / 'valid.txt', 'r') as f:
#         raw_val = []
#         for line in f.readlines():
#             raw_val.append(line.strip("\n").split(","))

#     # Get uniques
#     statement_entities, statement_predicates = _get_uniques_(raw_trn, raw_tst, raw_val)

#     st_entities = ['__na__'] + statement_entities
#     st_predicates = ['__na__'] + statement_predicates

#     entoid = {pred: i for i, pred in enumerate(st_entities)}
#     prtoid = {pred: i for i, pred in enumerate(st_predicates)}

#     train, valid, test = [], [], []
#     for st in raw_trn:
#         id_st = []
#         for i, uri in enumerate(st):
#             id_st.append(entoid[uri] if i % 2 == 0 else prtoid[uri])
#         train.append(id_st)
#     for st in raw_val:
#         id_st = []
#         for i, uri in enumerate(st):
#             id_st.append(entoid[uri] if i % 2 == 0 else prtoid[uri])
#         valid.append(id_st)
#     for st in raw_tst:
#         id_st = []
#         for i, uri in enumerate(st):
#             id_st.append(entoid[uri] if i % 2 == 0 else prtoid[uri])
#         test.append(id_st)

#     if subtype != "triples":
#         if subtype == "quints":
#             max_len = 5
#         train, valid, test = (
#             _pad_statements_(train, max_len),
#             _pad_statements_(valid, max_len),
#             _pad_statements_(test, max_len),
#         )

#     if subtype == "triples" or subtype == "quints":
#         train, valid, test = remove_dups(train), remove_dups(valid), remove_dups(test)

#     return {"train": train, "valid": valid, "test": test, "n_entities": len(st_entities),
#             "n_relations": len(st_predicates), 'e2id': entoid, 'r2id': prtoid}


# def get_wd_mappings() -> Tuple[Mapping[str, int], Mapping[str, int]]:
#     """
#         Returns the starE mappings
#     """
#     dataset_names = ['wd50k']
#     dataset_subtypes = ["triples", "quints", "statements"]

#     wd50k_statements = load_clean_wd50k(dataset_names[0],
#                                         subtype=dataset_subtypes[2])

#     # Get the StarE mappings
#     entoid = wd50k_statements['e2id']
#     reltoid = wd50k_statements['r2id']

#     return entoid, reltoid


# def create_mapping() -> None:
#     """Running this file creates the mappings and stores them."""
#     # TODO it appears we can implement this in place rather easily. Butfor now just calling this.

#     entoid, reltoid = get_wd_mappings()

#     # storing these mappings. Also correcting the Q numbers to include the full URL
#     folder = mapping_root
#     folder.mkdir(parents=True, exist_ok=True)
#     reltoidFile = folder / "reltoid"
#     entoidFile = folder / "entoid"

#     # first store reltoid. This does not need any more things to be added as they cannot act as variables
#     sortedRelations = __sort_map_pairs_by_numeric_string_key(reltoid)
#     with open(reltoidFile, "w") as relFile:
#         for (propertyID, index) in sortedRelations:
#             fullpropertyURL = "https://www.wikidata.org/wiki/Property:" + propertyID
#             relFile.write(f"{fullpropertyURL}\t{index}\n")

#     with open(entoidFile, "w") as enFile:
#         # add all the entites
#         sortedEntities = __sort_map_pairs_by_numeric_string_key(entoid)
#         for (entityID, index) in sortedEntities:
#             if entityID == "__na__":
#                 # We skip this one.
#                 continue
#             assert not EntityMapper.is_valid_variable_name(entityID), f"Cannot use the entity {entityID}, because it can be confused with a variable"
#             assert not EntityMapper.get_target_entity_name == entityID, f"Cannot use the entity {entityID}, because it gets confused with the target {EntityMapper.get_target_entity_name}"
#             assert index > 0, f"Found an index > 0 (got {index}), indices must be positive numbers."  # note: it seems 0 could in principle be included. Butt hat could be the index of __na__ is we want to use such thing.
#             fullEntityURL = "https://www.wikidata.org/wiki/" + entityID
#             offSetIndex = index
#             enFile.write(f"{fullEntityURL}\t{offSetIndex}\n")


# # Lazy initialization
# _entity_mapper = _relation_mapper = None


# def get_relation_mapper() -> RelationMapper:
#     """Get the relation mapper. Create and load if necessary."""
#     global _relation_mapper
#     if _relation_mapper is not None:
#         return _relation_mapper

#     path = mapping_root.joinpath("reltoid")
#     if not path.is_file():
#         create_mapping()
#     _relation_mapper = RelationMapper(__load_mapping(path))
#     return _relation_mapper


# def get_entity_mapper() -> EntityMapper:
#     """Get the entity mapper. Create and load if necessary."""
#     global _entity_mapper
#     if _entity_mapper is not None:
#         return _entity_mapper

#     relation_mapper = get_relation_mapper()
#     path = mapping_root.joinpath("entoid")
#     assert path.is_file()
#     _entity_mapper = EntityMapper(__load_mapping(path), relation_mapper)
#     return _entity_mapper
