import pathlib
import re
from typing import Sequence

import click
from gqs import dataset_split


@click.group()
def main():
    """The main entry point."""


def validate_dataset_name(ctx, param: click.Option, value):
    assert param.name == "dataset"
    if value is None:
        raise click.MissingParameter("No dataset name specified")
    if re.match(r"[a-z_]+", value):
        return pathlib.Path(value)
    else:
        raise click.BadParameter(f"a dataset name must match [_a-z]+, got {value}")


option_dataset_name = click.option(
    "--dataset",
    type=str,
    help="The name of the dataset, this name will also be used for the folder and as a prefix in the database. Must match [_a-z]+",
    required=True,
    callback=validate_dataset_name,
)

option_seed = click.option(
    "--seed",
    type=int,
    help="The seed to be used for the random number generator.",
    default=0
)


@main.group()
def split():
    """Methods to split the original graph into a training, validation and test set"""


def _splits_from_path(dataset: pathlib.Path, train_f: float, validation_f: float, test_f: float) -> Sequence[dataset_split.Split]:
    splits_path = dataset / "splits"
    train = dataset_split.Split(train_f, splits_path / "train")
    validation = dataset_split.Split(validation_f, splits_path / "validation")
    test = dataset_split.Split(test_f, splits_path / "test")
    return [train, validation, test]


@split.command(name="random", help='''Create the data splits from the input file and store them in a directory with the same name as the dataset name. Each triple is randomly assignmed to a split''')
@click.option('--input', type=pathlib.Path, help='The original data file in n-triples format, lines starting with # are ignored', required=True)
@option_dataset_name
@option_seed
@click.option('--train-fraction', type=float, help='fraction of the triples to be put into the training set', default=0.70)
@click.option('--validation-fraction', type=float, help='fraction of the triples to be put into the validation set', default=0.10)
@click.option('--test-fraction', type=float, help='fraction of the triples to be put into the test set', default=0.20)
@click.option('--triple-count', type=int, help="The number of triples in the file. Providing this speeds up the process, butthe count must be exact.", default=None)
def split_random(input: pathlib.Path, dataset: pathlib.Path, seed: int, train_fraction: float, validation_fraction: float, test_fraction: float, triple_count: int):
    splits = _splits_from_path(dataset, train_fraction, validation_fraction, test_fraction)
    dataset_split.split_random(input, splits, seed, triple_count)


@split.command(name="round-robin", help='''Create the data splits from the input file and store them in a directory with the same name as the dataset name. Triples are assigned to splits in a round robin fashion.''')
@click.option('--input', type=pathlib.Path, help='The original data file in n-triples format, lines starting with # are ignored')
@option_dataset_name
@click.option('--train-fraction', type=float, help='fraction of the triples to be put into the training set', default=0.70)
@click.option('--validation-fraction', type=float, help='fraction of the triples to be put into the validation set', default=0.10)
@click.option('--test-fraction', type=float, help='fraction of the triples to be put into the test set', default=0.20)
def split_round_robin(input: pathlib.Path, dataset: pathlib.Path, train_fraction: float, validation_fraction: float, test_fraction: float):
    splits = _splits_from_path(dataset, train_fraction, validation_fraction, test_fraction)
    dataset_split.split_round_robin(input, splits)


@split.command(name="remove", help="Remove the split files for the given dataset")
@option_dataset_name
def remove_split(dataset: pathlib.Path):
    for split in ["train", "validation", "test"]:
        the_split = (dataset / "splits" / split)
        if the_split.exists():
            print(f"Removing {the_split.absolute()}")
            the_split.unlink(missing_ok=True)
        else:
            print(f"Could not find {the_split.absolute()}")


@main.command(name="put-splits-in-graphdb", help="Initialize the database and upload the different data splits")
@option_dataset_name
def initialize_database(dataset: pathlib.Path):
    pass


@main.command(name="clear-dataset-graphdb")
@option_dataset_name
def clear_database(dataset: pathlib.Path):
    pass
