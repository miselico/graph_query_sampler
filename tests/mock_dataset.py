import pathlib
from typing import Optional, Tuple
from gqs.dataset import Dataset
from gqs.mapping import EntityMapper, RelationMapper

__all__ = ["MockRelationMapper", "MockEntityMapper", "MockDataset"]


class MockRelationMapper(RelationMapper):
    def __init__(self, amount_predicates: int) -> None:
        predicates = ["predicate_" + str(i) for i in range(amount_predicates)]
        super().__init__(predicates)


class MockEntityMapper(EntityMapper):
    def __init__(self, amount_entities: int, relation_mapper: RelationMapper) -> None:
        entities = ["entity_" + str(i) for i in range(amount_entities)]
        super().__init__(entities, relation_mapper)


class MockDataset(Dataset):
    def __init__(self, tmp_path: pathlib.Path, enitity_mapper: Optional[EntityMapper] = None, relation_mapper: Optional[RelationMapper] = None, dataset_name: str = "dataset_name",) -> None:
        super().__init__(dataset_name)
        tmp_path = tmp_path
        self.tmp_path = pathlib.Path(tmp_path)
        self.relmap = relation_mapper or MockRelationMapper(128)
        self.entmap = enitity_mapper or MockEntityMapper(256, self.relmap)

    def location(self) -> pathlib.Path:
        """For testing we override the location to a tmp directory"""
        return (self.tmp_path / "datasets" / self.name).resolve()

    def get_mappers(self) -> Tuple[RelationMapper, EntityMapper]:
        return self.relmap, self.entmap
