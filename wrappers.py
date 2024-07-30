"""
process StataHelper commands for each parallel process by parsing the arguments in order into the wildcard * placeholders, or
by name irrespective of the order of the arguments

"""
from multiprocessing import Pool, cpu_count
from typing import Tuple, Dict, List
from utils import read_yaml, config_yaml, cartesian


class CarriagePrint:
    def __init__(self,lst):
        self.lst = lst

    def __str__(self):
        return "\nPlan:\n\n"+"\n".join(self.lst)+f"\n\n # items: {str(len(self.lst))}\n\n"


def carriage_print(func):
    def wrapper(*args, **kwargs):
        results = func(*args, **kwargs)
        if isinstance(results, list):
            res = [r['cmd'] for r in results]
            return CarriagePrint(results)
        return results
    return wrapper


def parallelize(func, iterable, maxcores=None, buffer=1, *args):
    """
    wrapper for Pool.map or Pool.starmap
    :param func:  function to be mapped
    :param iterable:  iterable to be mapped, either a tuple or a list
    :param maxcores: maximum number of cores to use
    :param buffer: number of cores to leave available
    :return: list of results
    """

    cores = len(iterable)
    if maxcores is not None and cores > maxcores:
        cores = maxcores
    cores = cpu_count-buffer if cores > cpu_count() and buffer is not None else cores

    if args:
        iterable = [(i, *args) for i in iterable]

    with Pool(cores) as p:
        if isinstance(iterable[0], tuple):
            return p.starmap(func, iterable)
        else:
            return p.map(func, iterable)


class Parameters:
    """
    import parameters from a dictionary, list, yaml file, or string and expand the cartesian product of the parameters
    """
    def __init__(self):
        self.params = None
    #  --------------------------- Parameters in Parameters ---------------------------

    def from_dict(self, params: Dict[str, List[str]]):
        """
        extract parameters from a dictionary and assert valid dtypes, then expand the cartesian product of the parameters
        :param params: dictionary of strings
        """
        self.params = params
        return self.params

    def from_yaml(self, yamlpath: str | None):
        """
        extract parameters from YAML and assert valid dtypes, then expand the cartesian product of the parameters
        :param yamlpath: path to yaml file or None. if None, use the parameters from the config file
        """
        params = read_yaml(yamlpath)
        self.params = self.from_dict(params)
        return self.params
