import pytest
from utils import *


def cartesian_dict_happy():
    """
    Test the cartesian function with a dictionary
    """
    args = {
        'a': [[1, 2], [3, 4]],
        'b': [3, 4]
    }
    assert cartesian(args) == [
        ([1, 2], 3),
        ([1, 2], 4),
        ([3, 4], 3),
        ([3, 4], 4)
    ]


def cartesian_list_happy():
    """
    Test the cartesian function with a list
    """
    args = [
        [1, 2],
        [3, 4]
    ]
    assert cartesian(args) == [
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4)
    ]


def cartesian_sad():
    """
    Test the cartesian function with a string
    """
    args = 'a string'
    with pytest.raises(ValueError):
        cartesian(args)


def read_yaml_happy():
    """
    Test the read_yaml function with a string
    """
    yamlpath = 'tests/resources/test.yaml'
    assert read_yaml(yamlpath) == {
        'quiet': False,
        'y': None,
        'x': None,
        'fe1': None,
        'fe2': None,
        'arg1': None
        }


def read_yaml_sad():
    """
    Test the read_yaml function with a string
    """
    yamlpath = 123
    with pytest.raises(ValueError) as exp:
        read_yaml(yamlpath)
        assert str(exp.value) == "expected type str or dict, got <class 'int'>"


def read_yaml_file_not_found():
    """
    Test the read_yaml function with a string
    """
    yamlpath = 'tests/resources/not_a_file.yaml'
    with pytest.raises(FileNotFoundError):
        read_yaml(yamlpath)


def config_yaml_happy():
    """
    Test the config_yaml function
    """
    assert config_yaml() == {
        'quiet': False,
        'y': None,
        'x': None,
        'fe1': None,
        'fe2': None,
        'arg1': None
        }


def read_keys_happy():
    """
    Test the read_keys function with a string
    """
    string = 'quiet'
    dictionary = {
        'quiet': False,
        'y': None,
        'x': None,
        'fe1': None,
        'fe2': None,
        'arg1': None
        }
    assert read_keys(string, dictionary) == False


def read_keys_sad():
    """
    Test the read_keys function with a string
    """
    string = 'not_in_dict'
    dictionary = {
        'quiet': False,
        'y': None,
        'x': None,
        'fe1': None,
        'fe2': None,
        'arg1': None
        }
    assert read_keys(string, dictionary) is None
