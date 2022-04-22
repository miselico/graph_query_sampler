import unittest

from gqs import sample_queries


class Test_assert_query_validity(unittest.TestCase):
    def test_underscore(self) -> None:
        with self.assertRaises(AssertionError):
            sample_queries.assert_query_validity(["_"])

    def test_underscore_2(self) -> None:
        with self.assertRaises(AssertionError):
            sample_queries.assert_query_validity(["o1_var"])

    def test_underscore_var_target(self) -> None:
        with self.assertRaises(AssertionError):
            sample_queries.assert_query_validity(["_var_targets"])

    def test_one_hop_target_instead_of_targets(self) -> None:
        with self.assertRaises(AssertionError):
            sample_queries.assert_query_validity(["s0", "p0", "o0_target", "diameter"])

    def test_share_with_s_and_p(self) -> None:
        with self.assertRaises(AssertionError):
            sample_queries.assert_query_validity(["s0_p_0", "o_0_targets"])

    def test_one_hop(self) -> None:
        sample_queries.assert_query_validity(["s0", "p0", "o0_targets", "diameter"])

    def test_three_in_shared_qual(self) -> None:
        sample_queries.assert_query_validity(["s0", "p0", "o0_o1_o2_targets",
                                              "s1", "p1",
                                              "s2", "p2",
                                              "qr0i0_qr1i1",
                                              "qv0i0_qv1i1_var",
                                              "diameter"])

    def test_three_in_shared_qual_literals(self) -> None:
        sample_queries.assert_query_validity(["s0", "p0", "o0_o1_o2_targets",
                                              "s1", "p1",
                                              "s2", "p2",
                                              "qr0i0_qr1i1",
                                              "qv0i0_qvl1i1_var",
                                              "diameter"])


if __name__ == '__main__':
    unittest.main()
