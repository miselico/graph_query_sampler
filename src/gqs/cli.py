import logging
import os
import pathlib
import shutil
from typing import Optional, Sequence

import click
from gqs import dataset_split, split_to_triple_store, sample_queries, mapping as gqs_mapping
from gqs.conversion import convert_all, protobuf_builder
from gqs.dataset import BlankNodeStrategy, Dataset, initialize_dataset, initialize_dataset_from_TSV
from gqs.export import zero_qual_queries_dataset_to_KGReasoning

from ._sparql_execution import execute_sparql_to_result_silenced

logger = logging.getLogger(__file__)


@click.group(context_settings={'show_default': True})
def main() -> None:
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


@main.group()
def init() -> None:
    """Ïnitialize the dataset with RDF or from a TSV file"""


@init.command(name="RDF", help="Initialize the dataset with RDF data")
@click.option('--input', type=pathlib.Path, help='The original data file in n-triples format, lines starting with # are ignored', required=True)
@option_dataset
@click.option('--blank-node-strategy', type=click.Choice(['RAISE', 'CONVERT', 'IGNORE'], case_sensitive=False), default='RAISE',
              help="""What to do when blank nodes are encountered?
RAISE: raises an Exception
CONVERT: converts blank nodes to URIs, this is done based on their
IGNORE: ignores all triples involving blank nodes, effectively removing the node from the graph
""")
def init_RDF(input: pathlib.Path, dataset: Dataset, blank_node_strategy: str = 'RAISE') -> None:
    strategy = BlankNodeStrategy[blank_node_strategy]
    assert strategy is not None
    initialize_dataset(input, dataset, strategy)


@init.command(name="TSV", help="Initialize the dataset with TSV data")
@click.option('--input', type=pathlib.Path, help='The original data file in TSV format, without any headers!', required=True)
@option_dataset
def init_TSV(input: pathlib.Path, dataset: Dataset) -> None:
    initialize_dataset_from_TSV(input, dataset)


@main.group()
def split() -> None:
    """Methods to split the original graph into a training, validation and test set"""


@split.command(name="from-link-prediction-style", help='''Take the data splits from a directory with 3 files for train, valid and test, where there are three whitespace separated columns, the first one has the subjects, second the predicate and the last on the objects.''')
@click.option('--input', type=pathlib.Path, help='The folder with three files (train, valid, test)  in link prediction dataset format', required=True)
@option_dataset
def splits_from_dataset_link_prediction_style(input: pathlib.Path, dataset: Dataset) -> None:
    dataset_split.from_dataset_link_prediction_style(input, dataset)


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
def split_random(dataset: Dataset, seed: int, train_fraction: float, validation_fraction: float, test_fraction: float, triple_count: int) -> None:
    splits = _split_common(dataset, train_fraction, validation_fraction, test_fraction)
    dataset_split.split_random(dataset, splits, seed, triple_count)


@split.command(name="round-robin", help='''Create the data splits from the input file and store them in a directory with the same name as the dataset name. Triples are assigned to splits in a round robin fashion.''')
@option_dataset
@click.option('--train-fraction', type=float, help='fraction of the triples to be put into the training set', default=0.70)
@click.option('--validation-fraction', type=float, help='fraction of the triples to be put into the validation set', default=0.10)
@click.option('--test-fraction', type=float, help='fraction of the triples to be put into the test set', default=0.20)
def split_round_robin(dataset: Dataset, train_fraction: float, validation_fraction: float, test_fraction: float) -> None:
    splits = _split_common(dataset, train_fraction, validation_fraction, test_fraction)
    dataset_split.split_round_robin(dataset, splits)


@split.command(name="remove", help="Remove the split files for the given dataset")
@option_dataset
def remove_split(dataset: Dataset) -> None:
    for the_split in [dataset.train_split_location(), dataset.validation_split_location(), dataset.test_split_location()]:
        if the_split.exists():
            print(f"Removing {the_split.absolute()}")
            the_split.unlink(missing_ok=True)
        else:
            print(f"Could not find {the_split.absolute()}")


@main.group()
def store() -> None:
    "Put the datset splits into a triple store"


option_database_url = click.option(
    '--database-url', type=str, help='The URL of the database', default="http://localhost:7200/"
)


@store.command(name="graphdb", help='''Initialize a repository in the database and upload all splits which are in the dataset directory.
     The repositoryID will be equal to the dataset name, the named graphs will be split:test, split:train, split:validation, split:train-validation, and split:all''')
@option_dataset
@option_database_url
def store_graphdb(dataset: Dataset, database_url: str) -> None:
    for split in [dataset.train_split_location(), dataset.validation_split_location(), dataset.test_split_location()]:
        assert split.exists(), f"The split {split} could not be found, not starting"

    split_to_triple_store.create_graphdb_repository(dataset.graphDB_repositoryID(), database_url)
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
            split_to_triple_store.store_triples_graphDB(dataset, part, splitname, database_url)
    # check whether the imported files contains blank nodes, if so, warn the user
    query = """ASK {graph<split:all>{
            {?entity ?_p ?_o} UNION {?_s ?_p ?entity} UNION { <<?_s ?_p ?_o>> ?_qr ?entity }
        }
        FILTER(isBlank(?entity))
    }"""
    has_blank_nodes = execute_sparql_to_result_silenced(query, dataset.graphDB_url_to_endpoint(database_url), {})
    if has_blank_nodes:
        raise Exception("""You have imported data which contains blank nodes. This will lead to issues:
        Blank nodes will not be taken into account while generating the mapping to indices for tensors.
        Blank nodes in different splits will have different identities, causing query generation to fail.
        Do not continue, unless you know what you are doing.
        """)


@store.command(name="clear-graphdb", help="Remove the specified dataset from graphDB. Warning: all data in the dataset will be lost.")
@option_database_url
@option_dataset
def clear_database(dataset: Dataset, database_url: str) -> None:
    split_to_triple_store.remove_graphdb_repository(dataset, database_url)


@main.group()
def formulas() -> None:
    """Configure the formulas for the sampling. This includes copying suitable ones to the dataset directory, manually adding and removing what is needed.
         Then manually editing the config files as nessesary. Finally preprocessing the queries to include the constraints."""


@formulas.command(name="copy", help="Copy raw formulas to the dataset directory. These can then be changed.")
@option_dataset
@click.option("--formula-root", help="The root directory with the formulas to be copied to the dataset", type=pathlib.Path, required=True)
@click.option("--formula-glob", help="A glob pattern relative to --formula-root to select the fomulas to be copied, for patterns see https://docs.python.org/3.10/library/pathlib.html#pathlib.Path.glob", type=str, default="**/*")
@click.option("--force", is_flag=True, help="Continue even if there already are formulas in the dataset. This will overwrite without asking when the same files are copied again! Use with care.", type=bool, default=False)
def copy_formulas(dataset: Dataset, formula_root: pathlib.Path, formula_glob: str, force: bool) -> None:
    source_root = pathlib.Path(formula_root).resolve()
    target_root = dataset.raw_formulas_location()

    if not force and target_root.exists():
        if os.listdir(target_root):
            raise Exception("There are existing formulas in the dataset. Use --force to force merging and overwriting.")

    for source in source_root.glob(formula_glob):
        if source.is_file():
            relative_path = source.relative_to(source_root)
            target = target_root / relative_path
            target = target.resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


@formulas.command(name="add-constraints", help="Preprocess the formulas to add data specific restrictions")
@option_dataset
@option_database_url
def add_formula_constraints(dataset: Dataset, database_url: str) -> None:
    sparql_endpoint = dataset.graphDB_url_to_endpoint(database_url)
    sample_queries.preprocess_formulas(dataset, sparql_endpoint)


@main.group()
def sample() -> None:
    """Sample queries from the stored triples"""


@sample.command("create-graphdb")
@option_dataset
@option_database_url
def create_sample_graphdb(dataset: Dataset, database_url: str) -> None:
    """Create queries from the stored triples, store in CSV format."""
    sparql_endpoint = dataset.graphDB_url_to_endpoint(database_url)
    sample_queries.sample_queries(dataset, sparql_endpoint)


@sample.command("create-generic")
@option_dataset
@option_database_url
@click.option("--username", help="The username for basic HTTP authentication")
@click.option("--password", help="The password for basic HTTP authentication")
def create_sample_generic(dataset: Dataset, database_url: str, username: Optional[str], password: Optional[str]) -> None:
    """Create queries from triples stored in anzograph, store in CSV format."""
    sparql_endpoint = database_url or "http://localhost:8080/sparql"
    assert (username is None and password is None) or (username is not None and password is not None)
    if username is None:
        options = {}
    else:
        options = {"auth": (username, password)}
    sample_queries.sample_queries(dataset, sparql_endpoint, sparql_endpoint_options=options)


@sample.command("remove")
@option_dataset
def remove_sample(dataset: Dataset) -> None:
    """Delete the queries in CSV format."""
    sample_queries.remove_queries(dataset)


@main.group()
def mapping() -> None:
    """Manage numeric indices for identifiers used in the dataset. these are requird for converting queries into tensor forms"""


@mapping.command("create")
@option_dataset
@option_database_url
def create_mapping(dataset: Dataset, database_url: str) -> None:
    sparql_endpoint = dataset.graphDB_url_to_endpoint(database_url)
    gqs_mapping.create_mapping(dataset, sparql_endpoint, {})


@mapping.command("remove")
@option_dataset
def remove_mapping(dataset: Dataset) -> None:
    gqs_mapping.remove_mapping(dataset)


@main.group()
def convert() -> None:
    "Convert queries between forms"


@convert.command("csv-to-proto")
@option_dataset
def csv_to_proto(dataset: Dataset) -> None:
    """Convert the textual query results to index-based in protobuffer format.."""
    assert gqs_mapping.mapping_exists(dataset), "Before converting, you have to create a mapping"
    relmap, entmap = dataset.get_mappers()
    convert_all(
        source_directory=dataset.query_csv_location(),
        target_directory=dataset.query_proto_location(),
        builder=protobuf_builder(relmap=relmap, entmap=entmap)
    )


@main.group()
def export() -> None:
    "Export queries to other formats"


@export.command("to-kgreasoning")
@option_dataset
def csv_to_kgreasoning(dataset: Dataset) -> None:
    """Convert the queries into a format which can be parsed by KGreasoning."""
    zero_qual_queries_dataset_to_KGReasoning(dataset)
