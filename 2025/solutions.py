# %%
import re
from collections import namedtuple
from itertools import product, starmap
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

cat = "".join


def mapt(func, *args):
    return tuple(map(func, *args))


def ints(line):
    return mapt(int, line.split())


def quantify(iterable, pred=bool):
    return sum(pred(x) for x in iterable)


class Point(namedtuple("Point", ["i", "j"])):
    def __repr__(self):
        return f"({self.i}, {self.j})"


DIR4 = UP, RIGHT, DOWN, LEFT = tuple(
    starmap(Point, ((-1, 0), (0, 1), (1, 0), (0, -1)))
)
DIAGS = SE, NE, SW, NW = tuple(
    starmap(Point, ((1, 1), (-1, 1), (1, -1), (-1, -1)))
)
DIR8 = DIR4 + DIAGS


def add(p: Point, q: Point) -> Point:
    return Point(p.i + q.i, p.j + q.j)


def in_range(p: Point, N: int) -> bool:
    return (0 <= p.i < N) and (0 <= p.j < N)


def neighbors4(p: Point) -> tuple[Point, ...]:
    return tuple(add(p, d) for d in DIR4)


def neighbors8(p: Point) -> tuple[Point, ...]:
    return tuple(add(p, d) for d in DIR8)


def coords(N: int) -> tuple[Point, ...]:
    return tuple(starmap(Point, product(range(N), range(N))))


def at(grid, p: Point, default=None):
    return grid[p.i][p.j] if in_range(p, len(grid)) else default


lines = str.splitlines


def positive_integers(line):
    "get all positive integers from a line"
    return mapt(int, re.findall(r"\d+", line))


def Input(
    day, parser: Callable = str, sections: Callable = lines, is_test=False
):
    test = "test" if is_test else ""
    file_path = f"inputs/{day}{test}.txt"
    text = Path(file_path).read_text()
    return mapt(parser, sections(text))


# %%
################################################################################
# Day 1
################################################################################


# %%
# Part 1
def rotation(line):
    direction = 1 if line[0] == "R" else -1
    return direction * int(line[1:])


rots = Input(1, parser=rotation, is_test=False)
rots[:5]

# %%
positions = (np.cumsum(rots) + 50) % 100
quantify(positions == 0)

# %%
# Part 2
position = 50
zeros = 0
for rot in rots:
    dial = position + rot
    zeros += (
        (dial // 100)
        if rot >= 0
        else (position - 1) // 100 - (dial - 1) // 100
    )
    position = dial % 100
zeros

# %%
################################################################################
# Day 2
################################################################################

# %%
# Part 1
commas = lambda text: text.split(",")
endpoints = lambda section: mapt(int, section.split("-"))
ranges = Input(2, sections=commas, parser=endpoints, is_test=False)
ranges[:5]


# %%
def invalid_ids(lo, hi):
    ids = np.arange(lo, hi + 1)
    n_digits = np.ceil(np.log10(ids))
    divisors = 10 ** (n_digits // 2)
    front = ids // divisors
    back = ids % divisors
    return ids[front == back]


np.concat([invalid_ids(start, end) for start, end in ranges]).sum()


# %%
# Part 2
def flatten(iterable):
    return [item for sub in iterable for item in sub]


def first_half(n, repeat=2):
    digits = str(n)
    N = len(digits) // repeat
    divisible = len(digits) % repeat == 0
    return int(digits[:N]) if divisible else 10**N


def invalids(lo, hi, repeat=2):
    parts = range(first_half(lo, repeat), first_half(hi, repeat) + 1)
    return [
        id for part in parts if lo <= (id := int(str(part) * repeat)) <= hi
    ]


def invalids_in_range(lo, hi):
    digits = str(hi)
    return flatten(
        [
            invalids(lo, hi, repeat=repeat)
            for repeat in range(2, len(digits) + 1)
        ]
    )


# %%
assert first_half(1234, 2) == 12
assert first_half(3214, 4) == 3
assert first_half(12345, 2) == 100
assert first_half(3214, 3) == 10
assert invalids(998, 1000, 3) == [999]
assert invalids(2121212118, 2121212124, 5) == [2121212121]
assert invalids_in_range(998, 1000) == [999]
invalids_in_range(998, 1012)
# %%
sum(set(flatten(invalids_in_range(lo, hi) for lo, hi in ranges)))

# %%
################################################################################
# Day 3
################################################################################


# %%
def digits(n):
    return [int(d) for d in str(n)]


banks = Input(3, parser=digits, is_test=False)
len(banks)


# %%
def joltage(bank, n=2):
    if n == 1:
        return max(bank)
    sub = bank[: -n + 1]
    index = np.argmax(sub)
    return bank[index] * 10 ** (n - 1) + joltage(bank[index + 1 :], n - 1)


assert joltage([1, 2, 7]) == 27
assert joltage([1, 2, 3, 5, 1, 2, 9, 1, 8]) == 98
assert joltage(digits(234234234234278), 12) == 434234234278

# %%
# Part 1
sum(joltage(bank) for bank in banks)
# %%
# Part 2
sum(joltage(bank, 12) for bank in banks)

# %%
################################################################################
# Day 4
################################################################################

# %%
# Part 1
grid = Input(4, is_test=False)
grid[:5]


# %%
# Part 1
def accessible(grid, p: Point):
    return sum(at(grid, neighbor) == "@" for neighbor in neighbors8(p)) < 4


def all_accessible(grid):
    return [
        coord
        for coord in coords(len(grid))
        if at(grid, coord) == "@" and accessible(grid, coord)
    ]


len(all_accessible(grid))


# %%
# Part 2
def replace(grid, coords: Iterable[Point], value):
    """Replace the values of the grid at the coords with value"""
    grid_list = [list(row) for row in grid]
    for p in coords:
        if in_range(p, len(grid)):
            grid_list[p.i][p.j] = value
    return ["".join(row) for row in grid_list]


all_rolls = []
diagram = grid
while rolls := all_accessible(diagram):
    all_rolls.append(rolls)
    diagram = replace(diagram, rolls, ".")
sum(len(rolls) for rolls in all_rolls)

# %%
################################################################################
# Day 5
################################################################################

# %%
# Part 1
linebreak = lambda s: s.split("\n\n")
ranges, ids = Input(5, sections=linebreak, is_test=False)
ranges = mapt(positive_integers, lines(ranges))
ids = positive_integers(ids)
len(ranges), len(ids)


# %%
# Part 1
def fresh_ids(ranges, ids):
    return [id for id in ids if any(lo <= id <= hi for lo, hi in ranges)]


len(fresh_ids(ranges, ids))


# %%
# Part 2
def union_all(ranges):
    "finds non-overlapping ranges that cover the same area as the input ranges"
    sorted_ranges = sorted(ranges)
    result = [sorted_ranges[0]]

    for start, end in sorted_ranges[1:]:
        last_start, last_end = result[-1]
        # If current range overlaps with last range (start <= last_end), merge them
        if start <= last_end:
            result[-1] = (last_start, max(last_end, end))
        else:
            result.append((start, end))

    return tuple(result)


assert union_all(((1, 3), (2, 4), (5, 7))) == ((1, 4), (5, 7))

sum([hi - lo + 1 for lo, hi in union_all(ranges)])

# %%
################################################################################
# Day 6
################################################################################


# %%
def chars(text):
    return np.array(list(text))


def split_on_empty_rows(arr):
    """Split array into multiple arrays at rows where all elements are ' '"""
    split_indices = np.where(np.all(arr == " ", axis=1))[0]
    indices = np.concatenate(([-1], split_indices, [len(arr)]))
    for start, end in zip(indices[:-1], indices[1:]):
        if start + 1 < end:
            yield arr[start + 1 : end]


sheet = np.array(Input(6, parser=chars, is_test=False))
problems = list(split_on_empty_rows(sheet.T))
problems[0]


# %%
# Part 1
def compute(problem):
    tokens = mapt(cat, problem.T)
    nums = mapt(int, tokens[:-1])
    operator = tokens[-1].strip()
    return np.sum(nums) if operator == "+" else np.prod(nums)


sum(compute(problem) for problem in problems)


# %%
# Part 2
def cephalopod_compute(problem):
    tokens = mapt(cat, problem[:, :-1])
    nums = mapt(int, tokens)
    operator = cat(problem[:, -1]).strip()
    return np.sum(nums) if operator == "+" else np.prod(nums)


sum(cephalopod_compute(problem) for problem in problems)

# %%
################################################################################
# Day 7
################################################################################

grid = Input(7, is_test=True)
grid

# %%
# Part 1
