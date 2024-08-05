import pytest
from StataHelper import StataHelper
from wrappers import parallelize
from utils import limit_cores
import numpy as np
from multiprocessing import cpu_count
from builtins import *

params = {'y': ['mpg'], 'x': [['weight', 'length'], ['weight']]}
statapath = r"C:\Program Files\Stata18\utilities"
cmd = "regress {y} {x}"
estimatesdir = "D:\Collin\statahelper"
s = StataHelper(stata_path=statapath, edition='mp', splash=False,  set_output_dir=estimatesdir)
s.schedule(cmd, params)

def add(x, y):
    return x + y

def test_paralleize():
    assert parallelize(add, [(1, 2), (3, 4), (5, 6)]) == [3, 7, 11]

def test_parallelize_args():
    def p(func,iterable, *args):
        if args:
            iterable = [(i, *args) for i in iterable]
        return iterable
    assert p(add, [(1, 2), (3, 4), (5, 6)], 1) == [((1,2),1), ((3,4),1), ((5,6),1)]

def test_params_kwargs():
    kwargs = {'quetly': True}
    params = [(i, kwargs) for i in s.queue]
    assert params == [("regress mpg weight length", {'quetly': True}),
                              ("regress mpg weight", {'quetly': True})]

def test_params_no_kwargs():
    params = [i for i in s.queue]
    assert params == ["regress mpg weight length",
                              "regress mpg weight"]

def test_limit_maxcores():
    it = np.arange(1, 10)
    assert limit_cores(it, 2) == 2

def test_limit_big_iterable():
    it = np.arange(0, 100)
    assert limit_cores(it) == cpu_count()-1

def test_limit_small_iterable():
    it = np.arange(0, 4)
    assert limit_cores(it) == 4

def test_parallel_name():
    cmd = "regress {y} {x}\nestimates store *_{y}_{x}"
    pmap = {'y': ['mpg'], 'x': [['weight', 'length'], ['weight']]}
    name = 'test'
    kwargs = {'quetly': True}
    queue = s.schedule(cmd, pmap)
    params = list(enumerate([i for i in queue]))
    params = [(i, j) for i, j in params]
    assert params == [(0, 'regress mpg weight length'), (1, 'regress mpg weight')]

def test_parallel_name_kwargs():
    cmd = "regress {y} {x}\nestimates store *_{y}_{x}"
    pmap = {'y': ['mpg'], 'x': [['weight', 'length'], ['weight']]}
    name = 'test'
    kwargs = {'quetly': True}
    queue = s.schedule(cmd, pmap)
    params = list(enumerate([(i, kwargs) for i in queue]))
    params = [(i, j[0], j[1]) for i, j in params]
    assert params == [(0, 'regress mpg weight length', {'quetly': True}),
                      (1, 'regress mpg weight', {'quetly': True})]