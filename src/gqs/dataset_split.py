"""Functions to split a file with triples into a train, validation, and test split"""

import pathlib
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Callable, Iterable, MutableMapping, Set, TextIO

import pyhash


@dataclass
class Split:
    fraction: float
    path: pathlib.Path


def _hasher(splits: Iterable[Split], seed: int) -> Callable[[str], Split]:
    hashfunction = pyhash.city_32(seed)

    def f(line: str) -> Split:
        hashvalue = hashfunction(line)
        in_interval = hashvalue / float(2**32)
        current_accumulated = 0.0
        for split in splits:
            current_accumulated += split.fraction
            if in_interval <= current_accumulated:
                return split
        raise Exception("The flow should never reach here with a correct set of splits. Was validate splits called before the call?")
    return f


def split_hash(input_file: pathlib.Path, splits: Iterable[Split], seed: int):
    validate_splits(splits)
    # The following esures all files will be closed correctly
    with ExitStack() as stack:
        files_for_splits: MutableMapping[pathlib.Path, TextIO] = {}
        for split in splits:
            opened_file = stack.enter_context(open(split.path, "w"))
            files_for_splits[split.path] = opened_file
        hasher = _hasher(splits, seed)
        with open(input_file) as input:
            for line in input:
                line = line.strip()
                if line.startswith("#"):
                    continue
                selected_split = hasher(line)
                file = files_for_splits[selected_split.path]
                file.write(line + '\n')


def split_round_robin(input_file: pathlib.Path, splits: Iterable[Split]):
    # The following esures all files will be closed correctly
    counters: Counter[pathlib.Path] = Counter()
    total_count = 0
    with ExitStack() as stack:
        files_for_splits: MutableMapping[pathlib.Path, TextIO] = {}
        for split in splits:
            opened_file = stack.enter_context(open(split.path, "w"))
            files_for_splits[split.path] = opened_file
        with open(input_file) as input:
            for line in input:
                line = line.strip()
                if line.startswith("#"):
                    continue
                total_count += 1
                # check the counters one by one, if one of them is too low now, add it to that split
                for split in splits:
                    if total_count * split.fraction > counters[split.path]:
                        # add to this split
                        file = files_for_splits[split.path]
                        file.write(line + '\n')
                        counters[split.path] += 1
                        break


def validate_splits(splits: Iterable[Split]):
    total = 0.0
    paths: Set[pathlib.Path] = set()
    for split in splits:
        total += split.fraction
        if split.path in paths:
            raise Exception(f"Duplicate path '{split.path}' in the splits")
    assert abs(1 - total) < 0.001, "The sum of the fractions must add up to 1"
