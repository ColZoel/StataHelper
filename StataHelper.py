"""
Stata: a simplified Python wrapper and parallelizer for Stata
"""
import sys
from typing import List, Tuple, Dict

from exceptions.validate import _DefaultMissing
from utils import *
from parallel import *
import pandas as pd
import numpy as np
from exceptions.validate import *

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


class Stata:
    # TODO: implement STATA in __init__ and wrappers
    # Todo: default to config.yaml?
    def __init__(self, config=None,
                 stata_path=None,
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
        self.data = None
        self.safety_buffer = 1
        self.maxcores = self.cores-self.safety_buffer
        self.results = None
        self.expected_params = None
        self.status = None
        self.is_stata_initialized = None

        if config is not None:
            self.config = read_yaml(config)
            self.stata_path = read_keys('stata_path', self.config)
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
            self.stata_path = stata_path
            self.edition = edition
            self.splash = splash
            self.set_graph_format = set_graph_format
            self.set_graph_size = set_graph_size
            self.set_graph_show = set_graph_show
            self.set_command_show = set_command_show
            self.set_autocompletion = set_autocompletion
            self.set_streaming_output = set_streaming_output
            self.set_output_file = set_output_file

        sys.path.append(self.stata_path)
        import pystata
        pystata.config.init(
            edition=self.edition,
            splash=self.splash,
        )
        if self.set_graph_format is not None:
            pystata.config.set_graph_format(self.set_graph_format)
        if self.set_graph_size is not None:
            pystata.config.set_graph_size(self.set_graph_size)
        if self.set_graph_show is not None:
            pystata.config.set_graph_show(self.set_graph_show)
        if self.set_command_show is not None:
            pystata.config.set_command_show(self.set_command_show)
        if self.set_autocompletion is not None:
            pystata.config.set_autocompletion(self.set_autocompletion)
        if self.set_streaming_output is not None:
            pystata.config.set_streaming_output(self.set_streaming_output)
        if self.set_output_file is not None:
            pystata.config.set_output_file(self.set_output_file)

        self.is_stata_initialized = pystata.config.is_stata_initialized()
        self.status = pystata.config.status()

        if not self.is_stata_initialized:
            raise ValueError("Stata is not initialized.")  # TODO: change to StataError

    def is_stata_initialized(self):
        """
        check if Stata is initialized: Wrapper for pystata.config.is_stata_initialized()
        """
        return self.is_stata_initialized

    def status(self):
        """
        check the status of the Stata instance. Wrapper for pystata.config.status()
        """
        return self.status

    @staticmethod
    def close_output_file():
        """
        close the output file. Wrapper for pystata.config.close_output_file()
        """
        return pystata.config.close_output_file()

    @staticmethod
    def run(cmd: str, *args, **kwargs):
        """
        run a single Stata command. wrapper for pystata.stata.run()
        :param cmd: Stata command
        """
        return pystata.stata.run(cmd, *args, **kwargs)

    @staticmethod
    def get_return() -> Dict:
        """
        get the return values from the last command
        wrapper for pystata.stata.get_return()
        """
        return pystata.stata.get_return()

    @staticmethod
    def get_ereturn() -> Dict:
        """
        get the e return values from the last command
        wrapper for pystata.stata.get_ereturn()
        """
        return pystata.stata.get_ereturn()


    @staticmethod
    def get_sreturn() -> Dict:
        """
        get the s return values from the last command
        wrapper for pystata.stata.get_sreturn()
        """
        return pystata.stata.get_sreturn()



    def use(self, path: np.array | str, frame: None | str = None, force=False, *args, **kwargs):
        """
        read any pandas supported file type and send to Stata instance
        """
        if isinstance(path, np.Array):
            self.data = path
            if frame is not None:
                pystata.stata.nparray_to_frame(self.data, frame, force=force)
            else:
                pystata.stata.nparray_to_data(self.data, force=force)
            return self

        elif isinstance(path, pd.DataFrame) or isinstance(path, pd.Series):
            self.data = path
        elif isinstance(path, str):
            extension = path.split('.')[-1]
            if extension == 'csv':
                self.data = pd.read_csv(path, *args, **kwargs)
            elif extension == 'xlsx':
                self.data = pd.read_excel(path, *args, **kwargs)
            elif extension == 'dta':
                self.data = pd.read_stata(path, *args, **kwargs)
            elif extension == 'parquet':
                self.data = pd.read_parquet(path, *args, **kwargs)
            elif extension == 'feather':
                self.data = pd.read_feather(path, *args, **kwargs)
            elif extension == 'sas':
                self.data = pd.read_sas(path, *args, **kwargs)
            elif extension == 'spss':
                self.data = pd.read_spss(path, *args, **kwargs)
            elif extension == 'html':
                self.data = pd.read_html(path, *args, **kwargs)
            elif extension == 'json':
                self.data = pd.read_json(path, *args, **kwargs)
            elif extension in ['pkl', 'pickle', 'tar', 'gz', 'bz2', 'xz', 'zip']:
                self.data = pd.read_pickle(path, *args, **kwargs)
            elif extension == 'sql':
                self.data = pd.read_sql(path, *args, **kwargs)
            elif extension == 'clipboard':
                self.data = pd.read_clipboard(*args, **kwargs)
            elif extension == 'xml':
                self.data = pd.read_xml(path, *args, **kwargs)
            else:
                raise ValueError(f"Unsupported file extension: {extension}. Check "
                                 f"https://pandas.pydata.org/docs/reference/io.html for supported file types.\n"
                                 f"Is your filetype supported by pandas but not listed here? Email zoellercollin@gmail.com"
                                 f"or open an issue on the Github repo to make it right.")
        else:
            raise ValueError("Unsupported file type. Array, Pandas objects, or saved files accepted. Check "
                             "https://pandas.pydata.org/docs/reference/io.html for supported file types.\n")

        if frame is not None:
            pystata.stata.pdataframe_to_frame(self.data, frame, force=force)
        else:
            pystata.stata.pdataframe_to_data(self.data, force=force)

        return self

    def save(self, path: str, frame: str | None = None,
             var: str | int | list = None,
             obs:  str | int | list =None,
             selectvar:  str | int | list=None,
             valuelabel: bool =False, missingval: str = _DefaultMissing(), *args, **kwargs):
        """
        save the data to a file. Will save automatically to `path' depending on the path's extension.
        Except `path`, all params are the same as `pystata.stata.data_to_pdataframe`.
        https://www.stata.com/python/pystata18/stata.html#pystata.stata.pdataframe_from_data
        :param path: path to save the file
        :param frame: name of the stata frame to save
        :param var: list of variables to save
        :param obs: list of observations to save
        :param selectvar: Observations for which selectvar!=0 will be selected
        :param valuelabel: Use value label
        :param missingval: string to replace missing values
        :param args: additional arguments
        :param kwargs: additional keyword arguments
        :return: None
        """
        extension = path.split('.')[-1]
        if extension is None or extension == '':
            raise ValueError("No file extension provided. Check "
                             "https://pandas.pydata.org/docs/reference/io.html for supported file types.\n")

        if frame is not None:
            self.data = pystata.stata.frame_to_pdataframe(frame, var=var, obs=obs, selectvar=selectvar,
                                                           valuelabel=valuelabel, missingval=missingval)
        else:
            self.data = pystata.stata.data_to_pdataframe(var=var, obs=obs, selectvar=selectvar,
                                                         valuelabel=valuelabel, missingval=missingval)

        if extension == 'csv':
            self.data.to_csv(path, *args, **kwargs)
        elif extension == 'xlsx':
            self.data.to_excel(path, *args, **kwargs)
        elif extension == 'dta':
            self.data.to_stata(path, *args, **kwargs)
        elif extension == 'parquet':
            self.data.to_parquet(path, *args, **kwargs)
        elif extension == 'feather':
            self.data.to_feather(path, *args, **kwargs)
        elif extension == 'sas':
            self.data.to_sas(path, *args, **kwargs)
        elif extension == 'spss':
            self.data.to_spss(path, *args, **kwargs)
        elif extension == 'html':
            self.data.to_html(path, *args, **kwargs)
        elif extension == 'json':
            self.data.to_json(path, *args, **kwargs)
        elif extension in ['pkl', 'pickle', 'tar', 'gz', 'bz2', 'xz', 'zip']:
            self.data.to_pickle(path, *args, **kwargs)
        elif extension == 'sql':
            self.data.to_sql(path, *args, **kwargs)
        elif extension == 'clipboard':
            self.data.to_clipboard(*args, **kwargs)
        elif extension == 'xml':
            self.data.to_xml(path, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {extension}. Check "
                             f"https://pandas.pydata.org/docs/reference/io.html for supported file types.\n"
                             f"Is there a filetype you would like to see supported (pandas or not)? create a new issue "
                             f"on the Github repo or email zoellercollin@gmail.com.")

        return self

    def parallel(self, cmd: str,
                 iterable: str | Dict | List[str | tuple | List[str]],
                 stata_wildcard: bool = False,
                 maxcores: int = None,
                 safety_buffer: int = 1,
                 show_batches: bool = False):
        """
        run a Stata command in parallel: wrapper for pystata.stata.Run() on multiple cores
        :param cmd: Template of Stata command
        :param iterable: arguments to be parallelized. Can come from a dictoinary, yaml, list of strings or tuples
        :param stata_wildcard: Treat all * as a stata wildcard e.g. 'drop y_*' or 'esttab *'
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

        if not stata_wildcard:
            self.expected_params = cmd.count("{") + cmd.count("*")
        else:
            self.expected_params = cmd.count("{")

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
    Stata().parallel("reg {y} {x}", {'y': ['mpg'], 'x': [['weight', 'length'], ['weight']]}, maxcores=2, safety_buffer=1, show_batches=True)
