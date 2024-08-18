"""
wrappers for Stata commands and functions

"""
from multiprocessing import Pool
from builtins import *
from multiprocessing import cpu_count


class CarriagePrint:
    def __init__(self, lst):
        self.lst = lst

    def __str__(self):
        return "\nPlan:\n\n"+"\n".join(self.lst)+f"\n\n # items: {str(len(self.lst))}\n\n"


def carriage_print(func):
    def wrapper(*args, **kwargs):
        results = func(*args, **kwargs)
        if isinstance(results, list):
            # res = [r['cmd'] for r in results]
            return CarriagePrint(results)
        return results
    return wrapper


def parallelize(func, iterable, cores=None, *args):
    """
    wrapper for Pool.map or Pool.starmap
    :param func:  function to be mapped
    :param iterable:  iterable to be mapped, either a tuple or a list
    :param cores:  number of cores to use
    :return: list of results
    """
    if cores is None:
        cores = len(iterable) if len(iterable) < cpu_count() else cpu_count()
    if args:
        iterable = [(i, *args) for i in iterable]

    with Pool(cores) as p:
        if isinstance(iterable[0], tuple):
            return p.starmap(func, iterable)
        else:
            return p.map(func, iterable)
