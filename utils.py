"""
Tools for internal code processing
"""
import itertools

import yaml
import os

from itertools import product
from typing import List, Tuple, Dict
from wrappers import Parameters


class _DefaultMissing:
    '''
    Stata class. Here for replication.
    https://www.stata.com/python/pystata18/stata.html#ref-defaultmissing
    '''
    def __repr__(self):
        return "_DefaultMissing()"


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


def get_params(params: Dict[str, List[str]]):
    if isinstance(params, str):
        if params.split('.')[-1] not in ['yaml', 'yml']:
            raise ValueError("Expected a yaml  or yml file")
        return Parameters().from_yaml(params).args
    elif isinstance(params, dict):
        return Parameters().from_dict(params).args
    else:
        raise TypeError(f"Unsupported iterable type {type(params)}."
                        " Expected a yaml file, dictionary, list of strings, or list of tuples.")