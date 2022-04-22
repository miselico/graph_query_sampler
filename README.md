Graph Query Sampler (gqs)
========================

Graph query Sampler provides an implementation to extract queries from a graph. This is used to train and evaluate approximate graph query answering (also called multi-hop reasoning) systems.

To install, clone and run

    pip install -e .

To run test install the test dependencies using

    pip install -e .[test]

and then execute the tests with

    pytest




## Compilation of the protocol buffer file

* Download the protocol buffer binary. We used 3.20 and have the same version in setup.cfg. Most likely it is possible to use a newer version and put a corresponding newer version of the python package.
* `protoc-3.20.0-linux-x86_64/bin/protoc  -I=./src/gqs/query_represenation/ --python_out=./src/gqs/query_represenation/ --pyi_out=./src/gqs/query_represenation/ ./src/gqs/query_represenation/query.prot`

Then, the version used above did generate stubs which mypy on the github CI complains about. Somehow it does not process the exclude directives correctly. Hence, some Mapping types without parameters were changed to Mapping[Any,Any]

