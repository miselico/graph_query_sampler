"""Tests for converting to a stripped (no qualifier) query."""
import pathlib
from typing import List

import gqs.dataset_split as dataset_split
import pytest
from gqs.dataset import Dataset


class MockDataset(Dataset):
    def __init__(self, tmp_path: pathlib.Path) -> None:
        super().__init__("dataset_name")
        self.tmp_path = pathlib.Path(tmp_path)

    def location(self) -> pathlib.Path:
        """For testing we override the location to a tmp directory"""
        return (self.tmp_path / "datasets" / self.name).resolve()


@pytest.fixture(autouse=True)
def create_input(tmp_path: pathlib.Path) -> None:
    dataset = MockDataset(tmp_path)
    dataset.raw_location().mkdir(parents=True, exist_ok=True)
    with open(dataset.raw_location() / "input.txt", "w") as inputfile:
        # adding content
        for i in range(10000):
            inputfile.write(f"S{i} P{i} O{i} .\n")


def get_split(split: dataset_split.Split) -> List[str]:
    with open(split.path) as file:
        return file.readlines()


def split_n_equal_random(dataset: Dataset, n: int) -> List[dataset_split.Split]:
    fraction = 1.0 / n
    splits = [dataset_split.Split(fraction, dataset.splits_location() / f"split{i}") for i in range(n)]
    dataset_split.split_random(dataset, splits, seed=0)
    return splits


def test_one_random_split(tmp_path: pathlib.Path) -> None:
    """Test 'splitting' all into the same split by randomization'."""
    # input_file = tmp_path / "input.txt"
    dataset = MockDataset(tmp_path)
    splits = split_n_equal_random(dataset, 1)
    data = get_split(splits[0])
    assert len(data) == 10000


def test_three_equal_random_splits(tmp_path: pathlib.Path) -> None:
    """Test splitting all into three splits by randomization'."""
    dataset = MockDataset(tmp_path)
    splits = split_n_equal_random(dataset, 3)
    data = get_split(splits[0])
    assert len(data) == 3333
    data = get_split(splits[1])
    assert len(data) == 3333
    data = get_split(splits[2])
    assert len(data) == 3334


def test_three_unbalanced_random_splits(tmp_path: pathlib.Path) -> None:
    dataset = MockDataset(tmp_path)

    splits = [dataset_split.Split(0.2, dataset.splits_location() / "20"),
              dataset_split.Split(0.1, dataset.splits_location() / "10"),
              dataset_split.Split(0.7, dataset.splits_location() / "70")]
    dataset_split.split_random(dataset, splits, seed=0)
    assert len(get_split(splits[0])) == 2000
    assert len(get_split(splits[1])) == 1000
    assert len(get_split(splits[2])) == 7000


def split_n_equal_rr(dataset: Dataset, n: int) -> List[dataset_split.Split]:
    fraction = 1.0 / n
    splits = [dataset_split.Split(fraction, dataset.splits_location() / f"split{i}") for i in range(n)]
    dataset_split.split_round_robin(dataset, splits)
    return splits


def test_one_rr_split(tmp_path: pathlib.Path) -> None:
    """Test 'splitting' all into the same split round robin'."""
    dataset = MockDataset(tmp_path)
    splits = split_n_equal_rr(dataset, 1)
    data = get_split(splits[0])
    assert len(data) == 10000


def test_three_equal_rr_splits(tmp_path: pathlib.Path) -> None:
    """Test splitting all into three splits by randomization'."""
    dataset = MockDataset(tmp_path)
    splits = split_n_equal_rr(dataset, 3)
    data = get_split(splits[0])
    assert len(data) == 3334
    data = get_split(splits[1])
    assert len(data) == 3333
    data = get_split(splits[2])
    assert len(data) == 3333


def test_three_unbalanced_rr_splits(tmp_path: pathlib.Path) -> None:
    dataset = MockDataset(tmp_path)
    splits = [dataset_split.Split(0.2, dataset.splits_location() / "20"),
              dataset_split.Split(0.1, dataset.splits_location() / "10"),
              dataset_split.Split(0.7, dataset.splits_location() / "70")]
    dataset_split.split_round_robin(dataset, splits)
    assert len(get_split(splits[0])) == 2000
    assert len(get_split(splits[1])) == 1000
    assert len(get_split(splits[2])) == 7000
