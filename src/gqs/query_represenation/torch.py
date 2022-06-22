import dataclasses
from typing import Optional, Sequence
from ..types import LongTensor
from ..mapping import RelationMapper
import torch


@dataclasses.dataclass()
class TorchQuery:
    """This class represents a single query with pytorch tensors"""

    #: A tensor of edges. shape: (2, num_edges)
    #: The first row contains the heads, the second row the tails.
    edge_index: LongTensor

    # A tensor of edge types (i.e. relation IDs), shape: (num_edges,)
    edge_type: LongTensor

    # A tensor of shape (3, number_of_qualifier_pairs), one column for each qualifier pair.
    # The column contains (in this order) the id of the qualifier relation,
    # the id of the qualifier value, and the index of the corresponding edge
    qualifier_index: LongTensor

    # A tensor with the ids of all easy answers for this query, shape: (num_answers,)
    easy_targets: LongTensor

    # A tensor with the ids of all hard answers for this query, shape: (num_answers,)
    hard_targets: LongTensor

    # The longest shortest path between two nodes in this query graph, a scalar tensor.
    query_diameter: LongTensor

    # The query structure
    query_structure: str

    # A flag to indicate that the inverse relations are already included in the tensors.
    # This is to prevent accidentally calling withInverses twice
    inverses_already_set: bool = False

    def __post_init__(self) -> None:
        assert _check_shape(tensor=self.edge_index, expected_shape=(2, None))
        assert _check_shape(tensor=self.edge_type, expected_shape=(None,))
        assert _check_shape(tensor=self.qualifier_index, expected_shape=(3, None))
        assert _check_shape(tensor=self.easy_targets, expected_shape=(None,))
        assert _check_shape(tensor=self.hard_targets, expected_shape=(None,))
        assert _check_shape(tensor=self.query_diameter, expected_shape=tuple())

    def get_number_of_triples(self) -> int:
        """
        Get number of triples in the query

        Returns
        -------
        number_of_triples int
            The number of triples in the query
        """
        return self.edge_index.size()[1]

    def get_number_of_qualifiers(self) -> int:
        """
        Get number of qualifiers in the query

        Returns
        -------
        number_of_qualifiers int
            The number of qualifiers in the query
        """
        return self.qualifier_index.size()[1]

    def with_inverses(self, relmap: RelationMapper) -> "TorchQuery":
        """
        Gives you a copy of this TorchQuery which has inverses added.

        Raises
        ------
        AssertionError
            If this TorchQuery already contains inverses.
            You can check the inverses_already_set property to check,
            but usually you should know from the programming context.

        Returns
        -------
        TorchQuery
            A new TorchQuery object with the inverse edges (and accompanying qualifiers) set
        """
        assert not self.inverses_already_set
        number_of_riples = self.get_number_of_triples()
        number_of_qualifiers = self.get_number_of_qualifiers()
        # Double the space for edges, edge types and qualifier
        new_edge_index = torch.full((2, number_of_riples * 2), -1, dtype=torch.int)
        new_edge_type = torch.full((number_of_riples * 2,), -1, dtype=torch.int)
        new_qualifier_index = torch.full((3, number_of_qualifiers * 2), -1, dtype=torch.int)
        # copy old values
        new_edge_index[:, 0:number_of_riples] = self.edge_index
        new_edge_type[:, 0:number_of_riples] = self.edge_type
        new_qualifier_index[:, 0:number_of_qualifiers] = self.qualifier_index
        # add the inverse values
        new_edge_index[0, number_of_riples:] = self.edge_index[1]
        new_edge_index[1, number_of_riples:] = self.edge_index[0]
        for index, val in enumerate(self.edge_type):
            new_edge_type[number_of_riples + index] = relmap.get_inverse_of_index(val)
        # for the qualifiers, we first copy and then update the indices to the corresponding triples
        new_qualifier_index[:, number_of_qualifiers:] = self.qualifier_index
        new_qualifier_index[2, number_of_qualifiers:] += number_of_riples

        new_easy_targets = self.easy_targets
        new_hard_targets = self.hard_targets
        new_query_diameter = self.query_diameter

        return TorchQuery(new_edge_index, new_edge_type, new_qualifier_index, new_easy_targets, new_hard_targets,
                          new_query_diameter, inverses_already_set=True)


class ShapeError(RuntimeError):
    """An error raised if the shape does not match."""

    def __init__(self, actual_shape: Sequence[int], expected_shape: Sequence[Optional[int]]) -> None:
        """
        Initialize the error.

        :param actual_shape:
            The actual shape.
        :param expected_shape:
            The expected shape.
        """
        super().__init__(f"Actual shape {actual_shape} does not match expected shape {expected_shape}")


def _check_shape(
    tensor: torch.Tensor,
    expected_shape: Sequence[Optional[int]],
) -> bool:
    """Check the shape of a tensor."""
    actual_shape = tensor.shape
    if len(expected_shape) != len(actual_shape):
        raise ShapeError(actual_shape=actual_shape, expected_shape=expected_shape)

    for dim, e_dim in zip(actual_shape, expected_shape):
        if e_dim is not None and dim != e_dim:
            raise ShapeError(actual_shape=actual_shape, expected_shape=expected_shape)

    return True
