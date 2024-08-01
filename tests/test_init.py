from builtins import print
import unittest

from StataHelper import StataHelper
params = {'y': ['mpg'], 'x': [['weight', 'length'], ['weight']]}
statapath = r"C:\Program Files\Stata18\utilities"
s = StataHelper(stata_path=statapath, edition='mp', splash=False)

class MyTestCase(unittest.TestCase):
    def test_stata_init(self):
        assert s is not None


    def test_stata_init_path(self):
        assert s.stata_path == r"C:\Program Files\Stata18\utilities"


    def test_stata_init_path_error(self):
        with self.assertRaises(ValueError):
            s1= StataHelper(stata_path='C:/Program Files/Stata17/StataMP-64.exe',
                            edition='mp', splash=False)


    def test_is_initialized(self):
        assert s.is_stata_initialized() == True

