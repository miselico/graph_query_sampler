from typing import List
from gqs.conversion import _get_triple_and_qualifier_count_from_headers
import pytest

testdata = [
    (",s0,p0,diameter,o0_target-easy,o0_target-hard".split(","), 1, 0),
    (",s0,p0,s1,p1,diameter,o0_o1_target-easy,o0_1_target-hard".split(","), 2, 0),
    (",s0,p0,s1,p1,qr0i0,qv0i0,qr1i0,qv1i0 diameter,o0_o1_target-easy,o0_1_target-hard".split(","), 2, 2),
    # duplicating the subject to test whether it catches it.
    (",s0,s0,p0,s1,p1,qr0i0,qv0i0,qr1i0,qv1i0 diameter,o0_o1_target-easy,o0_1_target-hard".split(","), 2, 2),

]


@pytest.mark.parametrize("fields,expected_triples,expected_quals", testdata)
def test_count_triples_and_qualifiers_from_list(fields: List[str], expected_triples: int, expected_quals: int) -> None:
    triples, quals = _get_triple_and_qualifier_count_from_headers(fields)
    assert triples == expected_triples
    assert quals == expected_quals
