from builtins import *
import pytest
from StataHelper import StataHelper
from utils import cartesian

params1 = {'y': ['mpg'], 'x': [['weight', 'length'], ['weight']]}
params2 = {'y': ['mpg'], 'x': ['weight', 'length'], 'z': ['displacement']}
cmd1 = "regress {y} {x}"
cmd2 = "regress {y} {x} {z}"
statapath = r"C:\Program Files\Stata18\utilities"
s = StataHelper(stata_path=statapath, edition='mp', splash=False)

def test_notequal_sad1():
    with pytest.raises(ValueError):
        q = s.schedule(cmd2, params1)

    with pytest.raises(ValueError):
        q = s.schedule(cmd1, params2)

def test_cartesian():
    assert cartesian(params1.values()) == [('mpg', 'weight'), ('mpg', ['weight', 'length'])]

def test_process_map():
    cartesian_args = cartesian(params1.values())
    process_maps = [dict(zip(params1.keys(), c)) for c in cartesian_args]
    assert process_maps == [{'y': 'mpg', 'x': 'weight'}, {'y': 'mpg', 'x': ['weight', 'length']}]

def test_queue():
    cartesian_args = cartesian(params1.values())
    process_maps = [dict(zip(params1.keys(), c)) for c in cartesian_args]
    queue = [s._parse_cmd(cmd1, i) for i in process_maps]
    assert queue == ['regress mpg weight', 'regress mpg weight length']

def test_schedule():
    q = s.schedule(cmd1, params1)
    expected = ['regress mpg weight length', 'regress mpg weight']
    assert q == expected