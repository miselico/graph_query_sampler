Graph Query Sampler (gqs)
========================

Graph query Sampler provides an implementation to extract queries from a graph. This is used to train and evaluate approximate graph query answering (also called multi-hop reasoning) systems.

To install, clone the repository.

We recommend creating a virtual environment using conda.

`conda create --name gqs_env --python=3.10`

from the home of this repository:

`conda activate gqs_env`

and run:

    pip install -e .

To run test install the test dependencies using

`pip install -e .[test]`

For MacOS

`pip install -e '.[test]'`

and then execute the tests with

    pytest


## Creating a query dataset

To create a new query dataset, follow these steps. We assume a dataset named hp in which must be in n-triples format.

When using the command line tool, you can always see more information and options by adding `--help` to a command.

1. Install graphDB. You need to configure it with a lot of memory for the query sampler.
2. Initialize the folder for your dataset. Specify your nt file and the name you want to use for your dataset, which can only contain lowercase characters.
```bash
gqs init --input hp.nt --dataset hp --blank-node-strategy convert
```
This will create a new folder with the name of your dataset under the folder called datasets. All data related to the query sampling will be stored in that folder.

3. Split the dataset in train, validation and test. There are several options for the splitting, but  here we just do round-robin
```bash
gqs split round-robin --dataset hp
```

4. Store the splits in the triple store:
```bash
gqs store graphdb --dataset hp
```

5. Create the mapping for your dataset. This is the mapping between identifiers in the RDF file and indices which will be used in the tensor representations.
```bash
gqs mapping create --dataset hp
```

6. Configure the formulas you want to use for sampling.
Make sure that the formulas are adapted to what you need, check the shapes and configurations.
Then copy them as follows, the `--formula-root` argument specifies the directory with formulas, the glob pattern specifies the files within that directory.
```bash
gqs formulas copy --formula-root ./resources/formulas_example/ --formula-glob '**/0qual//**/*'  --dataset hp
```

7. Apply the constraints to the queries with:
```bash
gqs formulas add-constraints --dataset hp
```

8. Sample the queries from the triple store.
```bash
gqs sample create --dataset hp
```

9. To use the queries, we convert them to protocol buffers
```bash
gqs convert csv-to-proto --dataset hp
```

Done! Now the queries can be loaded with the provided data loader.

Alternatively, you could export the queries to a format which can be loaded by the KGReasoning framework: https://github.com/pminervini/KGReasoning/

```bash
gqs export to-kgreasoning --dataset hp
```
The result of the export will be placed in `/datasets/{datasetname}/export/kgreasoning` these files can then be put as a dataset in the KGReasoning framework.

## Compilation of the protocol buffer file

* Download the protocol buffer binary. We used 3.20 and have the same version in setup.cfg. Most likely it is possible to use a newer version and put a corresponding newer version of the python package.
* `protoc-3.20.0-linux-x86_64/bin/protoc  -I=./src/gqs/query_represenation/ --python_out=./src/gqs/query_represenation/ --pyi_out=./src/gqs/query_represenation/ ./src/gqs/query_represenation/query.prot`

Then, the version used above did generate stubs which mypy on the github CI complains about. Somehow it does not process the exclude directives correctly. Hence, some Mapping types without parameters were changed to Mapping[Any,Any]

