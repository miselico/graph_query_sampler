
from enum import Enum
import hashlib
from pathlib import Path
import re
from typing import Callable, List, MutableMapping, Optional, TextIO, Tuple
import logging
import gqs.mapping

logger = logging.getLogger(__file__)


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
        files = [f for f in self.raw_location().iterdir() if f.suffix == ".nt"]
        assert len(files) == 1, f"There is not exactly 1 .nt file in the raw data directory {self.raw_location()}, aborting"
        return files[0]

    def identifier_mapping_location(self) -> Path:
        return self.raw_location() / "blank_node_map.txt"

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

    def export_location(self) -> Path:
        return self.location() / "export"

    def export_kgreasoning_location(self) -> Path:
        return self.export_location() / "kgreasoning/"

    def __str__(self) -> str:
        return f"Dataset({self.name})"


class BlankNodeStrategy(Enum):
    RAISE = 1
    CONVERT = 2
    IGNORE = 3


def _intialize(input: Path, dataset: Dataset, line_handler: Callable[[int, str, TextIO], None]) -> None:
    """
    Create the dataset by copying and converting the data. Each line from the original file is given to the line_handler with arguments:
    line_handler(line_number, line, output_file)
    The line handler is responsible for checkign for errorsand writing the needed parts for this line to the file.
    """
    if dataset.location().exists():
        raise Exception(f"The directory for {dataset} already exists ({dataset.location()}), remove it first")
    dataset.location().mkdir(parents=True)
    # copy the dataset to the folder 'raw' under the dataset folder
    dataset.raw_location().mkdir(parents=True)
    output_file = dataset.raw_location() / input.name
    with open(input, 'rt') as open_input, open(output_file, 'wt') as open_output:
        for line_number, line in enumerate(open_input):
            line = line.strip()
            if line in ["", "\n", "\r\n"]:
                continue
            else:
                try:
                    line_handler(line_number, line, open_output)
                except Exception as e:
                    try:
                        output_file.unlink()
                    except Exception as e2:
                        # we tried to remove the files, but something went wrong. this might eb because the files do not exist
                        logger.warning("something went wrong trying to remove the created files.", exc_info=e2)
                        pass
                    raise Exception(f"Something went wrong handling line {line_number}. Output files have been removed.") from e


class _IRIhashCache:
    def __init__(self) -> None:
        """Creates a deterministic mapping between identifiers and IRIs"""
        self.cache: MutableMapping[str, str] = {}

    def get_iri(self, blank: str) -> str:
        """return an IRI representation for this blank node"""
        if blank in self.cache:
            return self.cache[blank]
        else:
            hashed = hashlib.sha1(blank.encode("UTF-8")).hexdigest()
            new_iri = f"<gqsblanks:{hashed}>"
            self.cache[blank] = new_iri
            return new_iri

    def has_entries(self) -> bool:
        return len(self.cache) > 0

    def write_mapping(self, to_file: Path) -> None:
        with open(to_file, "wt") as open_file:
            open_file.writelines([f"{k}\t{v}\n" for (k, v) in self.cache.items()])


def initialize_dataset(input: Path, dataset: Dataset, blank_node_strategy: BlankNodeStrategy) -> None:
    blank_node_cache = _IRIhashCache()

    def line_handler(line_number: int, line: str, output_file: TextIO) -> None:
        # split in 3 parts. Only the literal in the object position can contain a literal, which can have spaces in it.
        line = line.strip()
        if line.startswith('#'):
            return
        # strip of the trailing dot
        assert line.endswith(".")
        line = line[:-1]
        parts = [entity.strip() for entity in line.split(maxsplit=2)]
        blanks = [entity.startswith("_:") for entity in parts]
        if any(blanks):
            if blank_node_strategy == BlankNodeStrategy.RAISE:
                raise Exception(f'The input files had a blank node on line {line_number}, aborting. Perhaps specifiy to ignore or convert blank nodes.')
            elif blank_node_strategy == BlankNodeStrategy.IGNORE:
                msg = f'The input files had a blank node on line {line_number}, this line was ignored'
                logger.warn(msg)
                return
            elif blank_node_strategy == BlankNodeStrategy.CONVERT:
                # We take the blank node(s), hash each of them, and cretae new URLs with these hashes
                assert len(parts) == 3
                converted: List[str] = []
                if blanks[0]:
                    converted.append(blank_node_cache.get_iri(parts[0]))
                else:
                    converted.append(parts[0])
                # Blank nodes can occur in the subject and object only.
                if blanks[1]:
                    raise Exception(f"A blank node can never occur in the predicate position, this happens on line {line_number}")
                else:
                    converted.append(parts[1])
                if blanks[2]:
                    converted.append(blank_node_cache.get_iri(parts[2]))
                else:
                    converted.append(parts[2])
                line = f"{converted[0]} {converted[1]} {converted[2]} "
            else:
                raise AssertionError("Logic should bever reach here all enum cases should have been handled")
        output_file.write(line + ".\n")
    _intialize(input, dataset, line_handler)
    if blank_node_cache.has_entries:
        blank_node_cache.write_mapping(dataset.identifier_mapping_location())


def initialize_dataset_from_TSV(input: Path, dataset: Dataset) -> None:
    blank_node_cache = _IRIhashCache()

    def line_handler(line_number: int, line: str, output_file: TextIO) -> None:
        # split in 3 parts. This format only has a tsv with each line a triple.
        line = line.strip()
        # strip of the trailing dot
        parts = [entity.strip() for entity in line.split()]
        assert len(parts) == 3
        # We take the 3 identifier, hash each of them, and cretae new URLs with these hashes
        converted = [blank_node_cache.get_iri(part) for part in parts]
        line = f"{converted[0]} {converted[1]} {converted[2]} "
        output_file.write(line + ".\n")
    _intialize(input, dataset, line_handler)
    blank_node_cache.write_mapping(dataset.identifier_mapping_location())
