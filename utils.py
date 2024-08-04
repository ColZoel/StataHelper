"""
Tools for internal code processing
"""
import itertools
from builtins import *

import yaml
import os

from itertools import product
from typing import List, Tuple, Dict
# from wrappers import Parameters
from pathlib import Path
import re

class _DefaultMissing:
    '''
    Pystata class. Here for replication.
    https://www.stata.com/python/pystata18/stata.html#ref-defaultmissing
    '''
    def __repr__(self):
        return "_DefaultMissing()"


class OverwriteError(Exception):
    def __init__(self, filepath, numfiles):
        self.message = (f"OverwriteError: {filepath} already exists. This operation will remove all {numfiles} files"
        "and saves new ones. If this is intended, set overwrite=True.")
        super().__init__(self.message)


def sep(iterable):
    return " ".join(map(str,iterable))


def sep_var(estimate_file):
    return os.path.basename(estimate_file).split('_')[0]

def literal(s):
    return "{"+ s + "}"

def literal_search(s):
    return re.findall(r'\{(.+?)\}', s)


def is_pathlike(string: str):
    try:
        Path(string)
        return True
    except Exception:
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


def config_yaml():
    config = os.path.join(os.path.dirname(__file__), 'config.yaml')
    return read_yaml(config)


def read_keys(str, dict):
    if str in dict.keys():
        return dict[str]
    else:
        return None


def progress(text, i, count):
    i = i+1
    l= 20
    pct = (i/count)*100
    width = int(l*i//count)
    bar = '#'*width+" "*(l-width)
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