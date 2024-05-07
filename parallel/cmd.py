"""
process Stata commands for each parallel process by parsing the arguments in order into the wildcard * placeholders, or
by name irrespective of the order of the arguments

"""
from multiprocessing import Pool, cpu_count
from typing import Tuple

def paralellize(func, iterable, maxcores=None, buffer=1):
    """
    wrapper for Pool.map or Pool.starmap
    :param func:  function to be mapped
    :param iterable:  iterable to be mapped, either a tuple or a list
    :return: list of results
    """

    cores = len(iterable)
    if maxcores is not None and cores > maxcores:
        cores = maxcores
    cores = cpu_count-buffer if cores > cpu_count() else cores

    with Pool(cores) as p:
        if isinstance(iterable[0], tuple):
            return p.starmap(func, iterable)
        else:
            return p.map(func, iterable)


def cmd_process(show: bool, cmd: str, args: Tuple):
    """
    wrapper for running a Stata command
    :param cmd: Stata command
    :param show: print the command to the console
    :param args: arguments to be passed to the command
    :return: result of the command
    """

    if "*" in cmd and "{" in cmd:
        raise ValueError("Stata command can only have a wildcard * or bracketed {}, not both.")

    if "*" in cmd:
        count = cmd.count("*")

        if isinstance(args, dict):
            args = list(args.values())

        for arg in args:
            if isinstance(arg, list):
                cmd = cmd.replace(f"*", " ".join(arg), 1)
            else:
                cmd = cmd.replace(f"*", arg, 1)
        if show:
            print(cmd)
        return cmd

    elif "{" in cmd:
        # replace the bracketed arguments with values from the dictionary
        if not isinstance(args, dict):
            raise ValueError("Stata command with bracketed {} arguments must be provided a dictionary with keys "
                             "corresponding to the arguments in the command.\n"
                             "e.g. command: 'reg {y} {x}, cluster({cluster})' \n"
                             "dictionary: {'y': 'mpg', 'x': ['weight', 'length'], 'cluster': 'make'}")

        for key, value in args.items():
            if isinstance(value, list):
                cmd = cmd.replace(f"{{{key}}}", " ".join(value))
            else:
                cmd = cmd.replace(f"{{{key}}}", value)
        if show:
            print(cmd)
    else:
        raise ValueError("Stata command must have a wildcard * or bracketed {} argument to indicate parameters to vary."
                         "e.g. 'reg * *, cluster(*)' or 'reg {y} {x}, cluster({cluster})' provided a dictionary with "
                         "keys 'y' 'x' and 'cluster'.\n")
