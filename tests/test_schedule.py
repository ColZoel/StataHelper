import unittest
from StataHelper import StataHelper
params1 = {'y': ['mpg'], 'x': [['weight', 'length'], ['weight']]}
params2 = {'y': ['mpg'], 'x': ['weight', 'length'], 'z': ['displacement']}
cmd1 = "regress {y} {x}"
cmd2 = "regress {y} {x} {z}"
statapath = r"C:\Program Files\Stata18\utilities"
s = StataHelper(stata_path=statapath, edition='mp', splash=False)


class MyTestCase(unittest.TestCase):

    def test_notequal_sad1(self):

        with self.assertRaises(ValueError):
            q = s.schedule(cmd2, params1)

        with self.assertRaises(ValueError):
            q = s.schedule(cmd1, params2)

    def test_queue(self):
        q = s.schedule(cmd1, params1)
        expected = ['regress mpg weight length', 'regress mpg weight']
        self.assertEqual(q, expected)  # add assertion here


if __name__ == '__main__':
    unittest.main()
