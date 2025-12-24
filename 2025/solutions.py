# %%
from pathlib import Path
from typing import Callable

import numpy as np

cat = "".join


def mapt(func, *args):
    return tuple(map(func, *args))


def ints(line):
    return mapt(int, line.split())


def quantify(iterable, pred=bool):
    return sum(pred(x) for x in iterable)


lines = str.splitlines


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
def invalid_ids(start, end):
    ids = np.arange(start, end + 1)
    n_digits = np.ceil(np.log10(ids))
    divisors = 10 ** (n_digits // 2)
    front = ids // divisors
    back = ids % divisors
    return ids[front == back]


np.concat([invalid_ids(start, end) for start, end in ranges]).sum()
