import logging
import pathlib
import shutil
from typing import Sequence

import click
from gqs import dataset_split, split_to_triple_store, sample_queries
from gqs.dataset import Dataset

logger = logging.getLogger(__file__)


@click.group()
def main():
    """The main entry point."""


# def validate_dataset_name(ctx, param: click.Option, value) -> Dataset:
#     assert param.name == "dataset"
#     if value is None:
#         raise click.MissingParameter("No dataset name specified")
#     value = str(value)
#     valid, expl = dataset.valid_name(value)
#     if valid:
#         return Dataset(value)
#     else:
#         raise click.BadParameter(expl)


option_dataset = click.option(
    "--dataset",
    type=Dataset,
    help="The name of the dataset, this name will also be used for the folder and as a prefix in the database. Must match [_a-z]+",
    required=True,
    #    callback=validate_dataset_name,
)

option_seed = click.option(
    "--seed",
    type=int,
    help="The seed to be used for the random number generator.",
    default=0
)


@main.command(name="init", help="Initialize the datset directory")
@click.option('--input', type=pathlib.Path, help='The original data file in n-triples format, lines starting with # are ignored', required=True)
@option_dataset
def init(input: pathlib.Path, dataset: Dataset):
    # create the dataset folder
    if dataset.location().exists():
        raise Exception(f"The directory for {dataset} already exists ({dataset.location()}), remove it first")
    dataset.location().mkdir(parents=True)
    # copy the dataset to the folder 'raw' under the dataset folder
    dataset.raw_location().mkdir(parents=True)
    shutil.copy2(str(input), dataset.raw_location())
    # create other directories
    dataset.splits_location().mkdir(parents=True)


@main.group()
def split():
    """Methods to split the original graph into a training, validation and test set"""


def _split_common(dataset: Dataset, train_fraction: float, validation_fraction: float, test_fraction: float
                  ) -> Sequence[dataset_split.Split]:
    train = dataset_split.Split(train_fraction, dataset.train_split_location())
    validation = dataset_split.Split(validation_fraction, dataset.validation_split_location())
    test = dataset_split.Split(test_fraction, dataset.test_split_location())
    return [train, validation, test]


@split.command(name="random", help='''Create the data splits from the input file and store them in a directory with the same name as the dataset name. Each triple is randomly assignmed to a split''')
@option_dataset
@option_seed
@click.option('--train-fraction', type=float, help='fraction of the triples to be put into the training set', default=0.70)
@click.option('--validation-fraction', type=float, help='fraction of the triples to be put into the validation set', default=0.10)
@click.option('--test-fraction', type=float, help='fraction of the triples to be put into the test set', default=0.20)
@click.option('--triple-count', type=int, help="The number of triples in the file. Providing this speeds up the process, butthe count must be exact.", default=None)
def split_random(dataset: Dataset, seed: int, train_fraction: float, validation_fraction: float, test_fraction: float, triple_count: int):
    splits = _split_common(dataset, train_fraction, validation_fraction, test_fraction)
    dataset_split.split_random(dataset, splits, seed, triple_count)


@split.command(name="round-robin", help='''Create the data splits from the input file and store them in a directory with the same name as the dataset name. Triples are assigned to splits in a round robin fashion.''')
@option_dataset
@click.option('--train-fraction', type=float, help='fraction of the triples to be put into the training set', default=0.70)
@click.option('--validation-fraction', type=float, help='fraction of the triples to be put into the validation set', default=0.10)
@click.option('--test-fraction', type=float, help='fraction of the triples to be put into the test set', default=0.20)
def split_round_robin(dataset: Dataset, train_fraction: float, validation_fraction: float, test_fraction: float):
    splits = _split_common(dataset, train_fraction, validation_fraction, test_fraction)
    dataset_split.split_round_robin(dataset, splits)


@split.command(name="remove", help="Remove the split files for the given dataset")
@option_dataset
def remove_split(dataset: Dataset):
    for the_split in [dataset.train_split_location(), dataset.validation_split_location(), dataset.test_split_location()]:
        if the_split.exists():
            print(f"Removing {the_split.absolute()}")
            the_split.unlink(missing_ok=True)
        else:
            print(f"Could not find {the_split.absolute()}")


@main.group()
def store():
    "Put the datset splits into a triple store"


option_database_url = click.option(
    '--database-url', type=str, help='The URL of the database', default="http://localhost:7200/"
)


@store.command(name="graphdb", help='''Initialize a repository in the database and upload all splits which are in the dataset directory.
     The repositoryID will be equal to the dataset name, the named graphs will be split:test, split:train, split:validation, split:train-validation, and split:all''')
@option_dataset
@option_database_url
def store_graphdb(dataset: Dataset, database_url: str):
    repositoryID = dataset.graphDB_repositoryID()
#    dataset_location = pathlib.Path("datasets") / dataset
    for split in [dataset.train_split_location(), dataset.validation_split_location(), dataset.test_split_location()]:
        assert split.exists(), f"The split {split} could not be found, not starting"

    split_to_triple_store.create_graphdb_repository(repositoryID, database_url)
    for split_combo in [
        ("split:train", [dataset.train_split_location()]),
        ("split:validation", [dataset.validation_split_location()]),
        ("split:test", [dataset.test_split_location()]),
        ("split:train-validation", [dataset.train_split_location(), dataset.validation_split_location()]),
        ("split:all", [dataset.train_split_location(), dataset.validation_split_location(), dataset.test_split_location()]),
    ]:
        splitname = split_combo[0]
        splitparts = split_combo[1]
        for part in splitparts:
            split_to_triple_store.store_triples(repositoryID, part, splitname, database_url)


@store.command(name="clear-graphdb", help="Remove the specified dataset from graphDB. Warning: all data in the dataset will be lost.")
@option_database_url
@option_dataset
def clear_database(dataset: Dataset, database_url: str):
    split_to_triple_store.remove_graphdb_repository(dataset, database_url)


@main.group()
def formulas():
    """Configure the formulas for the sampling. This includes copying suitable ones to the dataset directory, manually adding and removing what is needed.
         Then manually editing the config files as nessesary. Finally preprocessing the queries to include the constraints."""


@formulas.command(name="copy", help="Copy raw formulas to the dataset directory. These can then be changed.")
@option_dataset
@click.option("--formulas", help="The directory with raw formulas to copy to this dataset", type=pathlib.Path, required=True)
def copy_formulas(dataset: Dataset, formulas: pathlib.Path):
    shutil.copytree(formulas, dataset.raw_formulas_location())


@formulas.command(name="add-constraints", help="Preprocess the formulas to add data specific restrictions")
@option_dataset
@option_database_url
def add_formula_constraints(dataset: Dataset, database_url: str):
    sparql_endpoint = dataset.graphDB_url_to_endpoint(database_url)
    sample_queries.preprocess_formulas(dataset, sparql_endpoint)


@main.command()
@option_dataset
@option_database_url
@click.option("--keep-uncompressed", is_flag=True, default=False)
def sample(dataset: Dataset, database_url: str, keep_uncompressed: bool):
    """Create queries from the stored triples"""
    sparql_endpoint = dataset.graphDB_url_to_endpoint(database_url)
    sample_queries.execute_queries(dataset, sparql_endpoint, compress=not keep_uncompressed)
