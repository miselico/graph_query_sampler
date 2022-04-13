"""Functions to split a file with triples into a train, validation, and test split"""

import pathlib
import random
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Iterable, MutableMapping, Optional, Set, TextIO


@dataclass
class Split:
    fraction: float
    path: pathlib.Path


def _comment_line(line: str) -> bool:
    return line.strip().startswith("#")


def _count_non_comment_lines(input_file: pathlib.Path) -> int:
    count = 0
    with open(input_file) as input:
        for line in input:
            if not _comment_line(line):
                count += 1
    return count


def split_random(input_file: pathlib.Path, splits: Iterable[Split], seed: int, lines: Optional[int] = None):
    validate_splits(splits)
    # The following esures all files will be closed correctly
    if lines is None:
        lines = _count_non_comment_lines(input_file)

    with ExitStack() as stack:
        randomized_splits = []
        for split in splits:
            total_for_this_split = int(split.fraction * lines)
            split.path.parent.mkdir(parents=True, exist_ok=True)
            opened_file = stack.enter_context(open(split.path, "w"))
            randomized_splits.extend([opened_file for i in range(total_for_this_split)])
        # it is possibel that due to rounding, randomizedsplits is not exactly as long as the file
        while len(randomized_splits) > lines:
            # remove the last element
            randomized_splits.pop
        while len(randomized_splits) < lines:
            # repeat the last element
            randomized_splits.append(randomized_splits[len(randomized_splits) - 1])
        assert len(randomized_splits) == lines
        rng = random.Random(seed)
        rng.shuffle(randomized_splits)
        with open(input_file) as input:
            for line in input:
                line = line.strip()
                if _comment_line(line):
                    continue
                # we take of the last element, this does not really matter, but implicitly reverses the randomized list!
                if len(randomized_splits) == 0:
                    raise Exception("The specified number fo lines is smaller than the actual number")
                file = randomized_splits.pop()
                file.write(line + '\n')
        assert len(randomized_splits) == 0, "The specified number of lines was bigger than the actual number"


def split_round_robin(input_file: pathlib.Path, splits: Iterable[Split]):
    # The following esures all files will be closed correctly
    validate_splits(splits)
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
                if _comment_line(line):
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
