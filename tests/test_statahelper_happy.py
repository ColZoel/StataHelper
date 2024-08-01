import pytest

from StataHelper import StataHelper


def test_stata_init():
    stata = StataHelper()
    assert stata is not None


def test_stata_init_path():
    stata = StataHelper('C:/Program Files/Stata17/StataMP-64.exe')
    assert stata.stata_path == 'C:/Program Files/Stata17/StataMP-64.exe'


def test_stata_init_path_error():
    with pytest.raises(ValueError) as exp:
        stata = StataHelper('C:/Program Files/Stata17/StataMP-64.exe')
    assert str(exp.value) == 'Stata not found at C:/Program Files/Stata17/StataMP-64.exe'

