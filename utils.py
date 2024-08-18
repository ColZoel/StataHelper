"""
Tools for internal code processing
"""
from builtins import *
import yaml
import os
from itertools import product
from typing import List, Tuple, Union
from pathlib import Path
import re


class _DefaultMissing:
    """
    Pystata class. Here for replication.
    https://www.stata.com/python/pystata18/stata.html#ref-defaultmissing
    """

    def __repr__(self):
        return "_DefaultMissing()"


class OverwriteError(Exception):
    def __init__(self, filepath, numfiles):
        self.message = (f"OverwriteError: {filepath} already exists. This operation will remove all {numfiles} files"
                        "and saves new ones. If this is intended, set overwrite=True.")
        super().__init__(self.message)


def sep(iterable):
    return " ".join(map(str, iterable))


def sep_var(estimate_file):
    return os.path.basename(estimate_file).split('_')[0]


def literal(s):
    return "{" + s + "}"


def literal_search(s):
    # noinspection RegExpRedundantEscape
    return re.findall(r"\{(.+?)\}", s)


def is_pathlike(string: Union[str, bytes]):
    try:
        Path(string)
        return True
    except TypeError:
        return False


def cartesian(args: List[any] | dict.values) -> List[Tuple[any]]:
    """
    create a cartesian product of the arguments
    :param args: list of strings or iterables to be expanded
    """
    return list(product(*args))


def read_yaml(yamlpath: str | dict) -> dict:
    """
    read a yaml file
    :param yamlpath: path to yaml file
    :return: dictionary
    """
    if isinstance(yamlpath, str):
        with open(yamlpath, 'r') as file:
            return yaml.safe_load(file)
    elif isinstance(yamlpath, dict):
        return yamlpath
    else:
        raise ValueError(f"expected type str or dict, got {type(yamlpath)}")


def read_keys(key, dictionary):
    if key in dictionary.keys():
        return dictionary[key]
    else:
        return None


def progress(text, i, count):
    i = i + 1
    bl = 20
    pct = (i / count) * 100
    width = int(bl * i // count)
    bar = '#' * width + " " * (bl - width)
    print(f"\r{i} of {count} |{bar}| {int(pct)}% | {text}", end="")
    if i == count:
        print("")
    return None


def limit_cores(iterable, maxcores=None, buffer=1) -> int:
    cores = len(iterable)
    if maxcores is not None and cores > maxcores:
        cores = maxcores
    cores = os.cpu_count() - buffer if cores > os.cpu_count() and buffer is not None else cores
    return cores
