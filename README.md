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
* `mkdir ./src/gqs/generated/`
* `protoc-3.20.0-linux-x86_64/bin/protoc  -I=./src/gqs/ --python_out=./src/gqs/generated/ ./src/gqs/query.proto`

