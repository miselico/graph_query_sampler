Formulas examples and config example

These formulas are created based on very common query patterns used in research papers on approximate query embedding.
There are both variants for queries with qualifiers (=edge information) and without.


These queries have constraints on the in-degrees of internal nodes of the query.
Namely, the in-degree of the variable and target node cannot be higher than 50.
This can be configured in the `config.json` file which is located with each query shape. See the `config_example.py` file for max in degree and max out degree constrain examples.

After copying these formulas to your dataset, you can freely make modifications to it, or remove and add patterns.
