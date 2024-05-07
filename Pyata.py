"""
Pyata: a simplified Python wrapper and parallelizer for Stata
"""
from typing import List, Tuple, Dict
from utils import *
from multiprocessing import cpu_count
from parallel.cmd import *
from itertools import product

class Import:

    def __init__(self):
        self.args = None
        self.params = None
    #  --------------------------- Import in Parameters ---------------------------

    def from_yaml(self, yamlpath: str | None):
        """
        extract parameters from YAML and assert valid dtypes, then expand the cartesian product of the parameters
        :param yamlpath: path to yaml file or None. if None, use the parameters from the config file
        """
        if yamlpath is None:
            params = config_yaml().values
        else:
            params = read_yaml(yamlpath).values

        self.args = cartesian(params)
        return self

    def from_dict(self, args: Dict[str, List[str]]):
        """
        extract parameters from a dictionary and assert valid dtypes, then expand the cartesian product of the parameters
        :param args: dictionary of strings
        """
        combos = cartesian(args.values())
        self.args = [dict(zip(args.keys(), c)) for c in combos]
        print(self.args)
        return self

    def from_list(self, args: List[str]):
        """
        extract parameters from a list and assert valid dtypes, then expand the cartesian product of the parameters
        :param args: list of strings
        """
        self.args = cartesian(args)
        return self

    def from_args(self, args: List[str | tuple[str]]):
        """
        extract parameters from a string and assert valid dtypes, then expand the cartesian product of the parameters
        :param args: string
        """

        self.args = args
        return self

    def show(self, display: bool = False):
        """
        display the parameters
        :param display: display the parameters
        :param args: list of tuples of parameters returned from the cartesian product
        """
        if display:
            for arg in self.args:
                print(arg)
        return self


class Results:

    def __init__(self):
        self.results = None

    def show(self, display: bool = False):
        """
        display the results
        :param display: display the results
        :param results: list of results returned from the parallelized Stata command
        """
        if display:
            for result in self.results:
                print(result)
        return self


class Pyata:
    # TODO: implement STATA in __init__ and wrappers
    def __init__(self, config=None,
                 edition=None,
                 splash=None,
                 set_graph_format=None,
                 set_graph_size=None,
                 set_graph_show=None,
                 set_command_show=None,
                 set_autocompletion=None,
                 set_streaming_output=None,
                 set_output_file=None):

        self.args = None
        self.cmd = None
        self.cores = cpu_count()
        self.safety_buffer = 1
        self.maxcores = self.cores-self.safety_buffer
        self.results = None
        self.expected_params = None

        if config is not None:
            self.config = read_yaml(config)
            self.edition = read_keys('edition', self.config)
            self.splash = read_keys('display_splash', self.config)
            self.set_graph_format = read_keys('set_graph_format', self.config)
            self.set_graph_size = read_keys('set_graph_size', self.config)
            self.set_graph_show = read_keys('set_graph_show', self.config)
            self.set_command_show = read_keys('set_command_show', self.config)
            self.set_autocompletion = read_keys('set_autocompletion', self.config)
            self.set_streaming_output = read_keys('set_streaming_output', self.config)
            self.set_output_file = read_keys('set_output_file', self.config)
        else:
            self.edition = edition
            self.splash = splash
            self.set_graph_format = set_graph_format
            self.set_graph_size = set_graph_size
            self.set_graph_show = set_graph_show
            self.set_command_show = set_command_show
            self.set_autocompletion = set_autocompletion
            self.set_streaming_output = set_streaming_output
            self.set_output_file = set_output_file

    @staticmethod
    def is_stata_initialized():
        """
        check if Stata is initialized: Wrapper for pystata.config.is_stata_initialized()
        """
        pass

    @staticmethod
    def status():
        """
        check the status of the Stata instance. Wrapper for pystata.config.status()
        """
        pass

    @staticmethod
    def close_output_file():
        """
        close the output file. Wrapper for pystata.config.close_output_file()
        """
        pass

    def run(self, cmd: str):
        """
        run a Stata command: wrapper for pystata.stata.run()
        :param cmd: Stata command
        """
        pass

    def parallel(self, cmd: str,
                 iterable: str | Dict | List[str | tuple],
                 maxcores: int = None,
                 safety_buffer: int = 1,
                 show_batches: bool = False):
        """
        run a Stata command in parallel: wrapper for pystata.stata.Run() on multiple cores
        :param cmd: Template of Stata command
        :param iterable: arguments to be parallelized. Can come from a dictoinary, yaml, list of strings or tuples
        :param maxcores: maximum number of cores to use
        :param safety_buffer: number of cores to leave free
        :param show_batches: display the batches as they are processed
        """
        if maxcores is not None:
            self.maxcores = maxcores
        if safety_buffer is not None:
            self.safety_buffer = safety_buffer
        self.cmd = cmd
        self.results = None
        self.expected_params = cmd.count("{") + cmd.count("*")

        if isinstance(iterable, str):
            # TODO: handle paths to yamls
            pass

        elif isinstance(iterable, dict):
            self.args = Import().from_dict(iterable).show(show_batches).args
        elif isinstance(iterable, list) and isinstance(iterable[0], str):
            self.args = Import().from_list(iterable).show(show_batches).args
        elif isinstance(iterable, tuple) and isinstance(iterable[0], tuple):
            self.args = Import().from_args(iterable).show(show_batches).args
        inputted_params = len(self.args[0])
        if self.expected_params != inputted_params:
            raise ValueError(f"Expected {self.expected_params} parameters, but received {inputted_params}.")

        iterable = [(show_batches, self.cmd, arg) for arg in self.args]
        # for tup in iterable:
        #     cmd_process(*tup)
        self.results = paralellize(cmd_process, iterable, maxcores, safety_buffer)  # Todo: implement STATA
        return self

if __name__ == '__main__':
    Pyata().parallel("reg {y} {x}", {'y': ['mpg'], 'x': [['weight', 'length'], ['weight']]}, maxcores=2, safety_buffer=1, show_batches=True)
