"""
Stata: a simplified Python wrapper and parallelizer for Stata
"""
import sys
from typing import List, Tuple, Dict
from utils import *
from utils import _DefaultMissing
from wrappers import *
import pandas as pd
import numpy as np


class Stata:
    def __init__(self,
                 params: None | str | Dict = None,
                 config: None | str = None,
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

        self.cores = cpu_count()
        self.safety_buffer = 1
        self.maxcores = self.cores-self.safety_buffer

        self.is_stata_initialized = None
        self.expected_params = None
        self.count = None
        self.results = None
        self.data = None
        self.params = get_params(params)
        self.args = None
        self.cmd = None
        self.que = None

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
        pystata.config.init(edition=self.edition,plash=self.splash)
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
            raise SystemError("Stata is not initialized.")  # TODO: change to StataError

    def is_stata_initialized(self):
        """
        check if Stata is initialized: Wrapper for pystata.config.is_stata_initialized()
        """
        return self.is_stata_initialized

    @staticmethod
    def status():
        """
        check the status of the Stata instance. Wrapper for pystata.config.status()
        """
        import pystata
        return pystata.config.status

    @staticmethod
    def close_output_file():
        """
        close the output file. Wrapper for pystata.config.close_output_file()
        """
        import pystata
        return pystata.config.close_output_file()

    @staticmethod
    def run(cmd: str, *args, **kwargs):
        """
        run a single Stata command. wrapper for pystata.stata.run()
        :param cmd: Stata command
        """
        import pystata
        return pystata.stata.run(cmd, *args, **kwargs)

    @staticmethod
    def get_return() -> Dict:
        """
        get the return values from the last command
        wrapper for pystata.stata.get_return()
        """
        import pystata
        return pystata.stata.get_return()

    @staticmethod
    def get_ereturn() -> Dict:
        """
        get the e return values from the last command
        wrapper for pystata.stata.get_ereturn()
        """
        import pystata
        return pystata.stata.get_ereturn()

    @staticmethod
    def get_sreturn() -> Dict:
        """
        get the s return values from the last command
        wrapper for pystata.stata.get_sreturn()
        """
        import pystata
        return pystata.stata.get_sreturn()

    def use(self, dta: str, columns: List[str] | None = None, obs: str | None = None, *args, **kwargs):
        """
        Inline method to use data in Stata
        :param dta: str dta path
        :param columns: list of columns to use when loading data
        :param obs: observations to load
        :return: None. Data is loaded into Stata
        """
        cmd = "use"
        if columns is not None:
            columns = " ".join(columns)
            cmd = cmd + columns
        if obs is not None:
            cmd = cmd + " in " + obs
        if columns is None or obs is None:
            cmd = cmd + " using"
        cmd = cmd + " " + dta
        self.run(cmd, *args, **kwargs)
        return self

    def use_file(self,
                 path: np.array | str,
                 frame: None | str = None,
                 force=False,
                 *args,
                 **kwargs):
        """
        read any pandas supported file type and send to Stata instance
        """
        if isinstance(path, np.ndarray):
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

    def conditions(self, keys: List[str], sep="&", name="if"):
        """
        create a condition string for Stata
        :param keys: list of keys
        :param sep: separator
        :param name: name of the condition
        :return: condition string
        """
        subdict = {k: self.params[k] for k in keys}
        combinations = cartesian(subdict.values())
        conditions = [f"{sep}".join(map(str, c)) if "" not in c else " ".join(map(str, c)) for c in combinations]
        conditions = [f" if {c}" if c != "" else "" for c in conditions]
        self.params[name] = conditions
        self.params = {k: v for k, v in self.params.items() if k not in keys}
        return self

    @staticmethod
    def parse_cmd(cmd: str, params: Dict):
        """
        parse elements of a Stata command and replace wildcards or bracketed arguments with values from the arguments
        :param cmd: str Stata command
        :param params: dict of arguments
        :return: result of the command
        """
        for key, value in params.items():
            if isinstance(value, list):
                cmd = cmd.replace(f"{{{key}}}", " ".join(value))
            else:
                cmd = cmd.replace(f"{{{key}}}", value)
        return cmd

    @carriage_print
    def schedule(self, cmd: str):
        """
        Return the que of commands to be run in parallel (cartesian product). Analogous to the parallel method, but
        does not execute the commands.
        :param cmd: str Stata command template
        :param iterable: str, dict, list of strings or tuples to be parallelized
        :param stata_wildcards: list(int) or int indicating the indices of wildcards to be treated as stata wildcards
        :return: list of commands to be run in parallel
        """

        if not isinstance(cmd, str):
            raise TypeError(f" Invalid Stata command. Expected a string, got type {type(cmd)}.")
        if cmd.count("{") != len(self.args[0]):
            raise ValueError(f"Expected {self.expected_params} parameters, but received {len(self.args[0])}.")

        self.cmd = cmd
        cartesian_args = cartesian(self.args)
        itemized = [dict(zip(self.args.keys, c)) for c in cartesian_args]

        self.que = [self.parse_cmd(cmd, i) for i in itemized]
        self.count = len(self.que)
        return self.que

    def parallel(self, cmd: str,
                 maxcores: int = None,
                 safety_buffer: int = 1):
        """
        run a Stata command in parallel: wrapper for pystata.stata.Run() on multiple cores
        :param cmd: Template of Stata command
        :param iterable: arguments to be parallelized. Can come from a dictoinary, yaml, list of strings or tuples
        :param stata_wildcards: list(int) or int indicating the indices of wildcards
        in the command that should be treated as a stata wildcard
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

        # --------------------------- Parameters in Parameters ---------------------------

        schedule = self.schedule(cmd)
        self.results = parallelize(self.run, schedule, maxcores, safety_buffer)
        return self


if __name__ == '__main__':
    params = {'y': ['mpg'], 'x': [['weight', 'length'], ['weight']]}

    s = Stata(params)
    s.que("regress {y} {x}")

