"""Tests for converting to a stripped (no qualifier) query."""
import pathlib
from typing import List

import pytest

import gqs.dataset_split as dataset_split


@pytest.fixture(autouse=True)
def create_input(tmp_path):
    with open(tmp_path / "input.txt", "w") as inputfile:
        # adding content
        for i in range(10000):
            inputfile.write(f"S{i} P{i} O{i} .\n")


def get_split(split: dataset_split.Split) -> List[str]:
    with open(split.path) as file:
        return file.readlines()


def split_n_equal_random(input_file: pathlib.Path, n: int, tmp_path: pathlib.Path):
    fraction = 1.0 / n
    splits = [dataset_split.Split(fraction, tmp_path / f"split{i}") for i in range(n)]
    dataset_split.split_random(input_file, splits, seed=0)
    return splits


def test_one_random_split(tmp_path):
    """Test 'splitting' all into the same split by randomization'."""
    input_file = tmp_path / "input.txt"
    splits = split_n_equal_random(input_file, 1, tmp_path)
    data = get_split(splits[0])
    assert len(data) == 10000


def test_three_equal_random_splits(tmp_path):
    """Test splitting all into three splits by randomization'."""
    input_file = tmp_path / "input.txt"
    splits = split_n_equal_random(input_file, 3, tmp_path)
    data = get_split(splits[0])
    assert len(data) == 3333
    data = get_split(splits[1])
    assert len(data) == 3333
    data = get_split(splits[2])
    assert len(data) == 3334


def test_three_unbalanced_random_splits(tmp_path):
    input_file = tmp_path / "input.txt"
    splits = [dataset_split.Split(0.2, tmp_path / "20"),
              dataset_split.Split(0.1, tmp_path / "10"),
              dataset_split.Split(0.7, tmp_path / "70")]
    dataset_split.split_random(input_file, splits, seed=0)
    assert len(get_split(splits[0])) == 2000
    assert len(get_split(splits[1])) == 1000
    assert len(get_split(splits[2])) == 7000


def split_n_equal_rr(input_file: pathlib.Path, n: int, tmp_path: pathlib.Path):
    fraction = 1.0 / n
    splits = [dataset_split.Split(fraction, tmp_path / f"split{i}") for i in range(n)]
    dataset_split.split_round_robin(input_file, splits)
    return splits


def test_one_rr_split(tmp_path):
    """Test 'splitting' all into the same split round robin'."""
    input_file = tmp_path / "input.txt"
    splits = split_n_equal_rr(input_file, 1, tmp_path)
    data = get_split(splits[0])
    assert len(data) == 10000


def test_three_equal_rr_splits(tmp_path):
    """Test splitting all into three splits by randomization'."""
    input_file = tmp_path / "input.txt"
    splits = split_n_equal_rr(input_file, 3, tmp_path)
    data = get_split(splits[0])
    assert len(data) == 3334
    data = get_split(splits[1])
    assert len(data) == 3333
    data = get_split(splits[2])
    assert len(data) == 3333


def test_three_unbalanced_rr_splits(tmp_path):
    input_file = tmp_path / "input.txt"
    splits = [dataset_split.Split(0.2, tmp_path / "20"),
              dataset_split.Split(0.1, tmp_path / "10"),
              dataset_split.Split(0.7, tmp_path / "70")]
    dataset_split.split_round_robin(input_file, splits)
    assert len(get_split(splits[0])) == 2000
    assert len(get_split(splits[1])) == 1000
    assert len(get_split(splits[2])) == 7000
