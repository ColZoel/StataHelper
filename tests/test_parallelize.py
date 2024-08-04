from builtins import *
import unittest
from wrappers import parallelize
from utils import limit_cores
import numpy as np
from StataHelper import StataHelper
from multiprocessing import cpu_count

params = {'y': ['mpg'], 'x': [['weight', 'length'], ['weight']]}
statapath = r"C:\Program Files\Stata18\utilities"
cmd = "regress {y} {x}"
estimatesdir = "D:\Collin\statahelper"
s = StataHelper(stata_path=statapath, edition='mp', splash=False,  set_output_dir=estimatesdir)
s.schedule(cmd, params)

def add(x, y):
    return x + y

class ParallelizeWrapper(unittest.TestCase):

    def test_paralleize(self):

        assert parallelize(add, [(1, 2), (3, 4), (5, 6)]) == [3, 7, 11]

    def test_parallelize_args(self):

        def p(func,iterable, *args):
            if args:
                iterable = [(i, *args) for i in iterable]
            return iterable

        assert p(add, [(1, 2), (3, 4), (5, 6)], 1) == [((1,2),1), ((3,4),1), ((5,6),1)]

class Test_SH_Parallel(unittest.TestCase):
    def test_params_kwargs(self):
        kwargs = {'quetly': True}
        params = [(i, kwargs) for i in s.queue]
        self.assertEqual(params, [("regress mpg weight length", {'quetly': True}),
                                          ("regress mpg weight", {'quetly': True})])

    def test_params_no_kwargs(self):
        params = [i for i in s.queue]
        self.assertEqual(params, ["regress mpg weight length",
                                          "regress mpg weight"])

class Test_limit_cores(unittest.TestCase):
    def test_limit_maxcores(self):
        it = np.arange(1, 10)
        assert limit_cores(it, 2) == 2

    def test_limit_big_iterable(self):
        it = np.arange(0, 100)
        assert limit_cores(it) == cpu_count()-1

    def test_limit_small_iterable(self):
        it = np.arange(0, 4)
        assert limit_cores(it) == 4


class TestTask_Names(unittest.TestCase):
    cmd = "regress {y} {x}\nestimates store *_{y}_{x}"

    def test_parallel_name(self, cmd, pmap, name, **kwargs):
        queue = s.schedule(cmd, pmap)
        params = list(enumerate([i for i in queue]))
        params = [(i, j) for i, j in params]
        assert params == [(0, 'regress mpg weight length'), (1, 'regress mpg weight')]

    def test_parallel_name_kwargs(self, cmd, pmap, name, **kwargs):
        queue = s.schedule(cmd, pmap)
        params = list(enumerate([(i, kwargs) for i in queue]))
        # remove the interior tuple
        params = [(i, j[0], j[1]) for i, j in params]
        assert params == [(0, 'regress mpg weight length', {'quetly': True}),
                          (1, 'regress mpg weight', {'quetly': True})]


if __name__ == '__main__':
    unittest.main()