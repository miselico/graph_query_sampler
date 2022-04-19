"""Tests for converting to a stripped (no qualifier) query."""

from gqs.mapping import EntityMapper, RelationMapper


def test_entity_mapper():
    for relation_count in range(1, 50, 7):
        relations = [f"relation{j}" for j in range(relation_count)]
        relmap = RelationMapper(relations)
        for entity_count in range(1, 200, 11):
            entities = [f"entity{j}" for j in range(entity_count)]
            entmap = EntityMapper(entities, relmap)

            # indices for normal entities
            entity_indices = set()
            for entity in entities:
                entindex = entmap.lookup(entity)
                assert entindex not in entity_indices, "There can be no reuse of indiced for different entities"
                entity_indices.add(entindex)
            assert len(entity_indices) == entity_count
            assert all(map(lambda index: isinstance(index, int), entity_indices))
            collected_indices = set(entity_indices)

            # reification nodes for relations
            predicate_entities = set()
            for relation in relations:
                predicate_entity = entmap.get_entity_for_predicate(relmap.lookup(relation))
                assert predicate_entity not in predicate_entities
                predicate_entities.add(predicate_entity)
            assert len(predicate_entities) == relation_count
            assert all(map(lambda index: isinstance(index, int), predicate_entities))
            assert len(collected_indices.intersection(predicate_entities)) == 0
            collected_indices = collected_indices.union(predicate_entities)

            # reification for inverse relations
            inverse_predicate_entities = set()
            for relation in relations:
                inverse_predicate_entity = entmap.get_entity_for_predicate(
                    relmap.get_inverse_of_index(relmap.lookup(relation)))
                assert inverse_predicate_entity not in inverse_predicate_entities
                inverse_predicate_entities.add(inverse_predicate_entity)
            assert len(inverse_predicate_entities) == relation_count
            assert all(map(lambda index: isinstance(index, int), inverse_predicate_entities))
            assert len(collected_indices.intersection(inverse_predicate_entities)) == 0
            collected_indices = collected_indices.union(inverse_predicate_entities)

            # reified statement indices
            for reified_statement in range(10):
                reified_statement_index = entmap.get_reified_statement_index(reified_statement)
                assert isinstance(reified_statement_index, int)
                assert reified_statement_index not in collected_indices
                assert entmap.is_entity_reified_statement(reified_statement_index)
                collected_indices.add(reified_statement_index)

            # variable indices
            for variable_name in [f"?var{i}" for i in range(0, 200, 5)]:
                assert entmap.is_valid_variable_name(variable_name)
                variable_index = entmap.lookup(variable_name)
                assert isinstance(variable_index, int)
                assert entmap.is_entity_variable(variable_index)
                assert variable_index not in collected_indices
                collected_indices.add(variable_index)

            # check target
            target_index = entmap.get_target_index()
            assert target_index == entmap.lookup(entmap.get_target_entity_name())
            assert target_index not in collected_indices
            assert entmap.is_entity_target(target_index)
            collected_indices.add(target_index)


def test_relation_mapper_one_relation():
    relmap = RelationMapper(["relation"])
    # all these assertions make assumptions regarding non-documented ordering of the indices.
    # users of the mapping must not make these assumptions
    assert relmap.lookup("relation") == 0
    assert relmap.get_inverse_of_index(0) == 4
    assert relmap.get_largest_forward_relation_id() == 3  # the one added and 3 for the reification relations
    assert relmap.reified_subject_index == 1
    assert relmap.reified_predicate_index == 2
    assert relmap.reified_object_index == 3


def test_relation_mapper_many_relations():
    for relation_count in range(1, 1000, 7):
        relations = [f"relation{j}" for j in range(relation_count)]
        relmap = RelationMapper(relations)
        # lookup all the relations:
        all_relation_indices = set()
        for relation in relations:
            relation_index = relmap.lookup(relation)
            assert relation_index not in all_relation_indices
            all_relation_indices.add(relation_index)
        assert len(all_relation_indices) == relation_count
        assert all(map(lambda index: isinstance(index, int), all_relation_indices))
        inverse_relation_indices = set()
        for relation_index in all_relation_indices:
            inverse_index = relmap.get_inverse_of_index(relation_index)
            assert inverse_index not in inverse_relation_indices
            inverse_relation_indices.add(inverse_index)
        assert len(inverse_relation_indices) == relation_count
        assert all(map(lambda index: isinstance(index, int), inverse_relation_indices))
        assert len(all_relation_indices.intersection(inverse_relation_indices)) == 0
        assert relmap.get_largest_forward_relation_id() == (relation_count - 1) + 3  # 3 for the reification relations
        reification_indices = set([relmap.reified_subject_index, relmap.reified_predicate_index, relmap.reified_object_index])
        assert len(reification_indices) == 3, "The values for the reification indices must be different"
        assert len(reification_indices.intersection(all_relation_indices.union(inverse_relation_indices))) == 0, "the reification indices must not coincide with any of the normal forward and inverse indices"
