from builtins import *
import unittest
from wrappers import parallelize
from utils import limit_cores
import numpy as np
from os import cpu_count

from StataHelper import StataHelper
params = {'y': ['mpg'], 'x': [['weight', 'length'], ['weight']], 'fes': ['make', 'foreign']}
statapath = r"C:\Program Files\Stata18\utilities"
cmd = "regress {y} {x}"
estimatesdir = "D:\Collin\statahelper"
s = StataHelper(stata_path=statapath, edition='mp', splash=False,  set_estimates_dir=estimatesdir)
s.schedule(cmd, params)

class ParallelizeWrapper(unittest.TestCase):

    def test_paralleize(self):
        def add(x, y):
            return x + y

        assert parallelize(add, [(1, 2), (3, 4), (5, 6)]) == [3, 7, 11]

    def test_arallelize_args(self):
        def add(x, y, z):
            return x + y + z

        assert parallelize(add, [(1, 2), (3, 4), (5, 6)], z=1) == [4, 8, 12]

    def test_maxcores(self):
        def add(x, y):
            return x + y

        assert parallelize(add, [(1, 2), (3, 4), (5, 6)], maxcores=2) == [3, 7, 11]

    def test_buffer(self):
        def add(x, y):
            return x + y

        assert parallelize(add, [(1, 2), (3, 4), (5, 6)], buffer=1) == [3, 7, 11]


class Test_add_output(unittest.TestCase):
    for p in params:
        cmd = s._parse_cmd(cmd, params)




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

        def test_cores_maxcores(self):
            params = list(np.zeros((100, 2)))
            cores = limit_cores(params, maxcores=4, buffer=1)
            self.assertEqual(cores, 4)

        def test_cores_no_maxcores(self):
            params = list(np.zeros((100, 2)))
            cores = limit_cores(params, buffer=1)
            self.assertEqual(cores, cpu_count()-1)


