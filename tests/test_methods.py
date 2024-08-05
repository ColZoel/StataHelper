import pytest
from builtins import ValueError

from StataHelper import StataHelper

params = {'y': ['mpg'], 'x': [['weight', 'length'], ['weight']]}
statapath = r"C:\Program Files\Stata18\utilities"
cmd = "regress {y} {x}\nestimates store *_{y}_{x}"
estimatesdir = "D:\Collin\statahelper"
dta = f'{statapath.replace("utilities", "auto.dta")}'

s = StataHelper(stata_path=statapath, edition='mp', splash=True,  set_output_dir=estimatesdir)

def test_load():
    s.use(dta)# add assertion here

def test_columns_list():
    s.use(dta, columns=['mpg', 'weight'])

def test_columns_string():
    s.use(dta, columns='mpg weight')

def test_columns_list_error():
    with pytest.raises(ValueError):
        s.use(dta, columns=1)

def test_obs1():
    s.use(dta, obs=10)


def test_obs2():
    s.use(dta, obs="10")


def test_obs3():
    s.use(dta, obs="1/10")


def test_obs_cols():
    s.use(dta, obs="1/10", columns=['mpg', 'weight'])




# Test functions that send data from pandas dataframes to Stata

def test_use_file():
    pass

def test_use_file_error():
    with pytest.raises(ValueError):
        s.use_file(dta)


def test_use_as_pandas():
    s.use(dta)
    df = s.use_as_pandas()
    assert df.shape == (74, 12)

def test_use_as_pandas_frame():
    s.use(dta)
    df = s.use_as_pandas(frame="newframe")
    assert df.shape == (74, 12)


def test_use_as_pandas_error():
    with pytest.raises(ValueError):
        s.use_as_pandas()


def test_save():
    s.save(dta)

def test_save_error():
    with pytest.raises(ValueError):
        s.save(1)
