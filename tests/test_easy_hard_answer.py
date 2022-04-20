
"""Tests for filtering hard and easy ansers for queries"""

import pandas as pd
from gqs.sample_queries import _merge_train_validation

from pandas.util.testing import assert_frame_equal


def test_merge_train_validation():
    train = pd.DataFrame({
        "s0": ["a", "b", "c", "q"],
        "p0": ["d", "e", "f", "r"],
        "o0_target": [set(["g", "h"]), set(["i", "j", "k"]), set(["l"]), {"s"}]
    })
    validation = pd.DataFrame({
        "s0": ["b", "c", "m", "q"],
        "p0": ["e", "f", "n", "r"],
        "o0_target": [set(["l", "k"]), set(["l", "p"]), set(["o"]), {"s"}]
        # note: the first case cannot really happen for a one hop.
        # The answer set would never overlap, but it could for more hops since the intermediate veriable can be different.
    })
    target_column = "o0_target"
    common_columns = ["s0", "p0"]
    result = _merge_train_validation(train, validation, target_column, common_columns)
    result = _stringify_sets(result, target_column)
    expected_result = _stringify_sets(pd.DataFrame({
        "s0": ["b", "c", "m"],
        "p0": ["e", "f", "n"],
        "o0_target_easy": [{"i", "j", "k"}, {"l"}, {}],
        "o0_target_hard": [{"l"}, {"p"}, {"o"}]
    }), target_column)

    return assert_frame_equal(result.sort_index(axis=1), expected_result.sort_index(axis=1), check_names=True)

    raise NotImplementedError("test not completed")


def _stringify_sets(merged: pd.DataFrame, target_column: str) -> pd.DataFrame:
    merged[target_column + "_hard"] = merged[target_column + "_hard"].map(lambda hardset: "|".join(hardset))
    merged[target_column + "_easy"] = merged[target_column + "_easy"].map(lambda easyset: "|".join(easyset) if isinstance(easyset, set) else "")
    return merged
