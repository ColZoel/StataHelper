import unittest
from StataHelper import StataHelper

params = {'y': ['mpg'], 'x': [['weight', 'length'], ['weight']]}
statapath = r"C:\Program Files\Stata18\utilities"
cmd = "regress {y} {x}\nestimates store *_{y}_{x}"
estimatesdir = "D:\Collin\statahelper"


s = StataHelper(stata_path=statapath, edition='mp', splash=True,  set_output_dir=estimatesdir)

class Test_Use(unittest.TestCase):

    # TODO: Set assert statements
    def test_load(self):
        s.use("auto.dta")# add assertion here

    def test_columns_list(self):
        s.use("auto.dta", columns=['mpg', 'weight'])

    def test_columns_string(self):
        s.use("auto.dta", columns='mpg weight')

    def test_columns_list_error(self):
        with self.assertRaises(ValueError):
            s.use("auto.dta", columns=1)

    def test_obs(self):
        s.use("auto.dta", obs=10)

class Test_use_file(unittest.TestCase):
    """
    Tests functions that send data from pandas dataframes to Stata
    """
    def test_use_file(self):
        pass

    def test_use_file_error(self):
        with self.assertRaises(ValueError):
            s.use_file("auto.dta")


class Test_use_as_pandas(unittest.TestCase):
    def test_use_as_pandas(self):
        s.use("auto.dta")
        df = s.use_as_pandas()
        self.assertEqual(df.shape, (74, 12))

    def test_use_as_pandas_frame(self):
        s.use("auto.dta")
        df = s.use_as_pandas(frame="newframe")
        self.assertEqual(df.shape, (74, 12))


    def test_use_as_pandas_error(self):
        with self.assertRaises(ValueError):
            s.use_as_pandas()

class Test_Save(unittest.TestCase):
    def test_save(self):
        s.save("auto.dta")

    def test_save_error(self):
        with self.assertRaises(ValueError):
            s.save(1)



if __name__ == '__main__':
    unittest.main()

