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
    StataHelper class. Here for replication.
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


class Parameters:
    """
    import parameters from a dictionary, list, yaml file, or string and expand the cartesian product of the parameters
    """
    def __init__(self):
        self.params = None
    #  --------------------------- Parameters in Parameters ---------------------------

    def from_dict(self, params: Dict[str, List[str]]):
        """
        extract parameters from a dictionary and assert valid dtypes, then expand the cartesian product of the parameters
        :param params: dictionary of strings
        """
        self.params = params
        return self.params

    def from_yaml(self, yamlpath: str | None):
        """
        extract parameters from YAML and assert valid dtypes, then expand the cartesian product of the parameters
        :param yamlpath: path to yaml file or None. if None, use the parameters from the config file
        """
        params = read_yaml(yamlpath)
        self.params = self.from_dict(params)
        return self.params


def get_params(params: Dict[str, List[str]]|str|None):
    if params is None:
        return None
    if isinstance(params, str):
        if params.split('.')[-1] not in ['yaml', 'yml']:
            raise ValueError("Expected a yaml  or yml file")
        return Parameters().from_yaml(params)
    elif isinstance(params, dict):
        return Parameters().from_dict(params)
    else:
        raise TypeError(f"Unsupported iterable type {type(params)}."
                        " Expected a yaml file, dictionary, list of strings, or list of tuples.")


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


def limit_cores(iterable, maxcores=None, buffer=1):
    cores = len(iterable)
    if maxcores is not None and cores > maxcores:
        cores = maxcores
    cores = os.cpu_count() - buffer if cores > os.cpu_count() and buffer is not None else cores
    return cores