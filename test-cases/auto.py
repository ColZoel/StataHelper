'''
Test Example to run a simple parallezied regression in Stata
'''

from StataHelper import StataHelper
params = {'y': ['mpg'], 'x': [['weight', 'length'], ['weight']]}
statapath = r"C:\Program Files\Stata18\utilities"
cmd = "regress {y} {x}\nestimates store *_{y}_{x}"
estimatesdir = "D:\Collin\statahelper"

if __name__ == '__main__':
    s = StataHelper(stata_path=statapath, edition='mp', splash=True,  set_output_dir=estimatesdir)
    s.run("use auto.dta", quietly=True)
    s.parallel(cmd, params, quietly=False)