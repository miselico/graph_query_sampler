import pathlib
import re

import click
from gqs import dataset_split


@click.group()
def main():
    """The main entry point."""


def validate_dataset_name(ctx, param, value):
    assert param == "dataset"
    if not re.match("[_a-z]+", value):
        return pathlib.Path(value)
    else:
        raise click.BadParameter("a dataset name must match [_a-z]+")


option_dataset_name = click.option(
    "--dataset",
    type=str,
    help="The name of the dataset, this name will also be used for the folder and as a prefix in the database. Must match [_a-z]+",
    callback=validate_dataset_name,
)

option_seed = click.option(
    "--seed",
    type=int,
    help="The seed to be used for the random number generator",
    default=0
)


@main.command(name="create-splits", help='''Create the data splits from the input file and store them in a directory with the same name as the dataset name''')
@click.option('--input', type=pathlib.Path, help='The original data file in n-triples format, lines starting with # are ignored')
@option_dataset_name
@option_seed
def split(input_file: pathlib.Path, dataset: pathlib.Path, seed: int):
    splits = dataset / "splits"
    train = splits / "train"
    validation = splits / "validation"
    test = splits / "test"
    dataset_split.split(input_file, train, validation, test, seed)


@main.command(name="clear-dataset-split", help="Remove the split files for the given dataset")
@option_dataset_name
def remove_split(dataset: pathlib.Path):
    pass


@main.command(name="put-splits-in-graphdb", help="Initialize the database and upload the different data splits")
@option_dataset_name
def initialize_database(dataset: pathlib.Path):
    pass


@main.command(name="clear-dataset-graphdb")
@option_dataset_name
def clear_database(dataset: pathlib.Path):
    pass
