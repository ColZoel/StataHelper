import pytest
from StataHelper import StataHelper
from builtins import ValueError

params = {'y': ['mpg'], 'x': [['weight', 'length'], ['weight']]}
statapath = r"C:\Program Files\Stata18\utilities"
s = StataHelper(stata_path=statapath, edition='mp', splash=False)

def test_stata_init():
    assert s is not None

def test_stata_init_path():
    assert s.stata_path == r"C:\Program Files\Stata18\utilities"

def test_stata_init_path_error():
    with pytest.raises(ValueError):
        s1= StataHelper(stata_path='C:/Program Files/Stata17/StataMP-64.exe',
                        edition='mp', splash=False)

def test_is_initialized():
    assert s.is_stata_initialized() == True

