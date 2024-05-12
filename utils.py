"""
Tools for internal code processing
"""
import itertools

import yaml
import os

from itertools import product
from typing import List, Tuple, Dict


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


def sys_yaml():
    system_config = os.path.join(os.path.dirname(__file__), 'sys_config.yaml')
    return read_yaml(system_config)


def config_yaml():
    config = os.path.join(os.path.dirname(__file__), 'config.yaml')
    return read_yaml(config)


def read_keys(str, dict):
    if str in dict.keys():
        return dict[str]
    else:
        return None