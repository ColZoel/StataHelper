from StataHelper import StataHelper
import os
from glob import glob
from time import perf_counter
def main():
# Run the Same stata code but in StataHelper to see the effect of the overhead
    with open('test-performance/test_gui_simple.do', 'r') as file:
        do_file = file.read()

    # Run the code in Stata
    start = perf_counter()
    s = StataHelper(stata_path = r"C:\Program Files\Stata18\utilities", edition='mp', splash=True)
    s.run(do_file, quietly=False)
    end = perf_counter()
    rounded =  round(end-start, 4)

    print(f"Time taken: {rounded} seconds")