import unittest
from StataHelper import StataHelper
params1 = {'y': ['mpg'], 'x': [['weight', 'length'], ['weight']]}
params2 = {'y': ['mpg'], 'x': ['weight', 'length'], 'z': ['displacement']}
cmd1 = "regress {y} {x}"
cmd2 = "regress {y} {x} {z}"
statapath = r"C:\Program Files\Stata18\utilities"
s = StataHelper(stata_path=statapath, edition='mp', splash=False)


class TestSchedule(unittest.TestCase):

    def test_notequal_sad1(self):

        with self.assertRaises(ValueError):
            q = s.schedule(cmd2, params1)

        with self.assertRaises(ValueError):
            q = s.schedule(cmd1, params2)

    def test_cartesian(self):
        from utils import cartesian
        assert cartesian(params1.values()) == [('mpg', 'weight'), ('mpg', ['weight', 'length'])]

    def test_process_map(self):
        from utils import cartesian
        cartesian_args = cartesian(params1.values())
        process_maps = [dict(zip(params1.keys(), c)) for c in cartesian_args]
        assert process_maps == [{'y': 'mpg', 'x': 'weight'}, {'y': 'mpg', 'x': ['weight', 'length']}]

    def test_queue(self):
        from utils import cartesian
        cartesian_args = cartesian(params1.values())
        process_maps = [dict(zip(params1.keys(), c)) for c in cartesian_args]
        queue = [s._parse_cmd(cmd1, i) for i in process_maps]
        assert queue == ['regress mpg weight', 'regress mpg weight length']


    def test_schedule(self):
        q = s.schedule(cmd1, params1)
        expected = ['regress mpg weight length', 'regress mpg weight']
        self.assertEqual(q, expected)  # add assertion here


if __name__ == '__main__':
    unittest.main()
