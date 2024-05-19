"""Tests for reification."""
import pathlib
import tempfile

from gqs.conversion import protobuf_builder
from gqs.loader import read_queries_from_proto_with_reification
from gqs.query_representation.torch import TorchQuery
from mock_dataset import MockDataset


def test_reification(tmp_path: pathlib.Path) -> None:

    dataset = MockDataset(tmp_path)
    relmap, entmap = dataset.get_mappers()
    builder = protobuf_builder(relmap, entmap)

    """Test creating a single query graph."""
    queryBuilder = builder(3, 3)
    queryBuilder.set_subject_predicate_entity_object(0,
                                                     "entity_1",
                                                     "predicate_1",
                                                     "?var1")
    queryBuilder.set_subject_predicate_entity_object(1,
                                                     "entity_1",
                                                     "predicate_5",
                                                     "?var1")
    queryBuilder.set_subject_predicate_entity_object(2,
                                                     "?var1",
                                                     "predicate_1",
                                                     entmap.get_target_entity_name())
    # def set_qualifier_rel_val(self, triple_index: int, qualifier_index: int, predicate: str, value: str) -> None:
    queryBuilder.set_qualifier_rel_entity_val(0, 0, "predicate_2", "entity_3")
    queryBuilder.set_qualifier_rel_entity_val(0, 1, "predicate_3", "entity_4")
    queryBuilder.set_qualifier_rel_entity_val(1, 2, "predicate_4", "entity_5")
    queryBuilder.set_diameter(2)
    queryBuilder.set_easy_entity_targets(["entity_6", "entity_7"])
    queryBuilder.set_hard_entity_targets(["entity_8"])
    theQuery = queryBuilder.build()
    with tempfile.TemporaryDirectory() as directory:
        # we use a normal file instead of a python temp file because a temp file is not guaranteed to be openable twice, which we need here
        proto_file_path = pathlib.Path(directory) / "queries_file.proto"
        queryBuilder.store([theQuery], proto_file_path)
        reified_queries: list[TorchQuery] = list(read_queries_from_proto_with_reification(proto_file_path, dataset=dataset))

        # start asserting stuff
        assert len(reified_queries) == 1
        reified_query = reified_queries[0]

        assert reified_query.edge_index.size(0) == 2
        assert reified_query.edge_index.size(1) == 12  # 3*3 for triple + 3 for qualifiers

        # all subjects must be variables
        subjects = reified_query.edge_index[0, :]
        for s in subjects:
            assert entmap.is_entity_reified_statement(int(s.item())), "All subject must be reified statement ids"
        for triple_nr in range(3):
            # Note: it is not strictly necessary that they appear in this order
            assert reified_query.edge_type[triple_nr * 3] == relmap.reified_subject_index
            assert reified_query.edge_type[triple_nr * 3 + 1] == relmap.reified_predicate_index
            assert reified_query.edge_type[triple_nr * 3 + 2] == relmap.reified_object_index
        # testing the reified triples
        assert reified_query.edge_index[1, 0] == entmap.lookup("entity_1")
        assert reified_query.edge_index[1, 1] == entmap.get_entity_for_predicate(relmap.lookup("predicate_1"))
        assert reified_query.edge_index[1, 2] == entmap.lookup("?var1")

        assert reified_query.edge_index[1, 3] == entmap.lookup("entity_1")
        assert reified_query.edge_index[1, 4] == entmap.get_entity_for_predicate(relmap.lookup("predicate_5"))
        assert reified_query.edge_index[1, 5] == entmap.lookup("?var1")

        assert reified_query.edge_index[1, 6] == entmap.lookup("?var1")
        assert reified_query.edge_index[1, 7] == entmap.get_entity_for_predicate(relmap.lookup("predicate_1"))
        assert reified_query.edge_index[1, 8] == entmap.get_target_index()

        # test whether the qualifiers have been correctly attached
        assert reified_query.edge_index[0, 9] == reified_query.edge_index[0, 0]
        assert reified_query.edge_index[0, 10] == reified_query.edge_index[0, 0]
        assert reified_query.edge_index[0, 11] == reified_query.edge_index[0, 3]

        # test whether the qualifiers ahve the correct realtion types and values
        assert reified_query.edge_type[9] == relmap.lookup("predicate_2")
        assert reified_query.edge_index[1, 9] == entmap.lookup("entity_3")
        assert reified_query.edge_type[10] == relmap.lookup("predicate_3")
        assert reified_query.edge_index[1, 10] == entmap.lookup("entity_4")
        assert reified_query.edge_type[11] == relmap.lookup("predicate_4")
        assert reified_query.edge_index[1, 11] == entmap.lookup("entity_5")

        assert len(reified_query.easy_targets) == 2
        assert reified_query.easy_targets[0] == entmap.lookup("entity_6")
        assert reified_query.easy_targets[1] == entmap.lookup("entity_7")

        assert len(reified_query.easy_targets) == 2
        assert reified_query.hard_targets[0] == entmap.lookup("entity_8")

        assert reified_query.query_diameter == 2
