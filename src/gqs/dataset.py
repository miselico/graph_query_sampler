
from pathlib import Path
import re
from typing import Optional, Tuple

import gqs.mapping


def valid_name(name: str) -> Tuple[bool, str]:
    if re.match(r"^[a-z0-9_]+$", name):
        return True, ""
    else:
        return False, f"a dataset name must match [_a-z]+, got {name}"


class Dataset:
    def __init__(self, dataset_name: str) -> None:
        valid, expl = valid_name(dataset_name)
        assert valid, expl
        self.name = dataset_name
        self._mappers: Optional[Tuple[gqs.mapping.RelationMapper, gqs.mapping.EntityMapper]] = None

    def location(self) -> Path:
        return (Path("./datasets") / self.name).resolve()

    def raw_location(self) -> Path:
        return self.location() / "rawdata/"

    def raw_input_file(self) -> Path:
        files = [f for f in self.raw_location().iterdir()]
        assert len(files) == 1, f"There is not exactly 1 file in the raw data directory {self.raw_location()}, aborting"
        return files[0]

    def splits_location(self) -> Path:
        return self.location() / "splits"

    def train_split_location(self) -> Path:
        return self.splits_location() / "train"

    def validation_split_location(self) -> Path:
        return self.splits_location() / "validation"

    def test_split_location(self) -> Path:
        return self.splits_location() / "test"

    def raw_formulas_location(self) -> Path:
        return self.location() / "formulas" / "raw"

    def formulas_location(self) -> Path:
        """The location of the preprocessed formulas"""
        return self.location() / "formulas" / "with_constraints"

    def query_location(self) -> Path:
        """The location where all queries in this dataset will be stored"""
        return self.location() / "queries"

    def raw_query_csv_location(self) -> Path:
        """The queries in CSV fromat with all answers for the split"""
        return self.query_location() / "raw_csv"

    def query_csv_location(self) -> Path:
        """The queries in CSV format including hard and easy answers"""
        return self.query_location() / "csv"

    def query_proto_location(self) -> Path:
        return self.query_location() / "proto"

    def mapping_location(self) -> Path:
        return self.location() / "mapping"

    def entity_mapping_location(self) -> Path:
        return self.mapping_location() / "entities.txt"

    def relation_mapping_location(self) -> Path:
        return self.mapping_location() / "relations.txt"

    def get_mappers(self) -> Tuple["gqs.mapping.RelationMapper", "gqs.mapping.EntityMapper"]:
        # lazy initialization
        if self._mappers is None:
            self._mappers = gqs.mapping.get_mappers(self)
        assert self._mappers is not None
        return self._mappers

    def graphDB_repositoryID(self) -> str:
        return "gqs-" + self.name

    def graphDB_url_to_endpoint(self, database_url: str) -> str:
        return database_url + "/repositories/" + self.graphDB_repositoryID()

    def __str__(self) -> str:
        return f"Dataset({self.name})"
