"""
StataHelper: a simplified Python wrapper and parallelizer for StataHelper
"""
import sys
from builtins import *
from utils import *
from utils import _DefaultMissing
from wrappers import *
import pandas as pd
import numpy as np
import os
import time
import datetime
from typing import Dict, List


# --------------------------- StataHelper Class ---------------------------
class StataHelper:
    def __init__(self,
                 stata_path=None,
                 edition=None,
                 splash=None,
                 set_output_dir=None,
                 set_graph_format=None,
                 set_graph_size=None,
                 set_graph_show=None,
                 set_command_show=None,
                 set_autocompletion=None,
                 set_streaming_output=None,
                 set_output_file=None):

        # --------------------------- Module Parameters ---------------------------
        self.input_dir = None
        self.output_dir = set_output_dir
        self.savename = ""
        self.cmd = None
        self.pmap = None
        self.queue = None
        self.qcount = None
        self.data = None
        # --------------------------- System/ Parallelization Parameters ---------------------------
        self.cores = os.cpu_count()
        self.safety_buffer = 1
        self.maxcores = self.cores-self.safety_buffer

        # --------------------------- Base Stata Configuration ---------------------------
        self.stata_path = stata_path
        self.edition = edition
        self.splash = splash
        self.graph_format = set_graph_format
        self.graph_size = set_graph_size
        self.graph_show = set_graph_show
        self.command_show = set_command_show
        self.autocompletion = set_autocompletion
        self.streaming_output = set_streaming_output
        self.output_file = set_output_file

        sys.path.append(self.stata_path)
        if not os.path.exists(self.stata_path):
            raise ValueError(f"Stata not found at {self.stata_path}")
        import pystata
        pystata.config.init(edition=self.edition, splash=self.splash)

        # TODO: test kwgs in each of these
        if self.graph_format is not None:
            pystata.config.set_graph_format(**self.graph_format)
        if self.graph_size is not None:
            pystata.config.set_graph_size(self.graph_size)
        if self.graph_show is not None:
            pystata.config.set_graph_show(self.graph_show)
        if self.command_show is not None:
            pystata.config.set_command_show(self.command_show)
        if self.autocompletion is not None:
            pystata.config.set_autocompletion(self.autocompletion)
        if self.streaming_output is not None:
            pystata.config.set_streaming_output(self.streaming_output)
        if self.output_file is not None:
            pystata.config.set_output_file(self.output_file)

        self.stata_initialized = pystata.config.is_stata_initialized()  # Doesn't seem to work in base Pystata module

        if not self.stata_initialized:
            raise SystemError("StataHelper is not initialized.")

    def is_stata_initialized(self):
        """
        check if StataHelper is initialized: Wrapper for pystata.config.is_stata_initialized()
        """
        return self.stata_initialized

    @staticmethod
    def status():
        """
        check the status of the StataHelper instance. Wrapper for pystata.config.status()
        """
        return pystata.config.status()

    @staticmethod
    def close_output_file():
        """
        close the output file. Wrapper for pystata.config.close_output_file()
        """
        import pystata
        return pystata.config.close_output_file()

    def run(self, cmd: str, *args, **kwargs):
        """
        run a single StataHelper command. wrapper for pystata.stata.run()
        :param cmd: StataHelper command
        """
        sys.path.append(self.stata_path)
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

    def use(self, dta: str, columns: List[str] | None = None, obs: str | None = None, **kwargs):
        """
        Inline method to use data in Stata
        :param dta: str dta path
        :param columns: list of columns to use when loading data
        :param obs: observations to load
        :return: None. Data is loaded into StataHelper
        """
        cmd = "use "
        if columns is not None:
            if isinstance(columns, list):
                columns = " ".join(columns)
            elif isinstance(columns, str):
                columns = columns
            else:
                raise ValueError("columns must be a list or string.")
            cmd = cmd + columns
        if obs is not None:
            if isinstance(obs, int):
                obs = str(obs)
            cmd = f"{cmd} in {obs}"
        if columns is not None or obs is not None:
            cmd = cmd + " using"
        cmd = cmd + f' "{dta}"'
        self.run(cmd, **kwargs)
        print(f"\n\n{dta} loaded.\n")
        return self

    def use_file(self,
                 path,
                 frame=None,
                 force=False,
                 *args,
                 **kwargs):
        """
        read any pandas supported file type and send to StataHelper instance
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
                                 f"Is your filetype supported by pandas but not listed here? "
                                 f"Email zoellercollin@gmail.com"
                                 f"or open an issue on the Github repo to make it right.")
        else:
            raise ValueError("Unsupported file type. Array, Pandas objects, or saved files accepted. Check "
                             "https://pandas.pydata.org/docs/reference/io.html for supported file types.\n")

        if frame is not None:
            pystata.stata.pdataframe_to_frame(self.data, frame, force=force)
        else:
            pystata.stata.pdataframe_to_data(self.data, force=force)

        return self

    @staticmethod
    def use_as_pandas(frame: str | None = None,
                      var: str | int | list = None,
                      obs: str | int | list = None,
                      selectvar: str | int | list = None,
                      valuelabel: bool = False,
                      missingval: str = _DefaultMissing()) -> pd.DataFrame:
        """
        return the data as a pandas dataframe.
        Wrapper for pystata.stata.frame_to_pdataframe and pystata.stata.data_to_pdataframe
        https://www.stata.com/python/pystata18/stata.html
        """

        if frame is not None:
            data = pystata.stata.frame_to_pdataframe(frame, var=var, obs=obs, selectvar=selectvar,
                                                     valuelabel=valuelabel, missingval=missingval)
        else:
            data = pystata.stata.data_to_pdataframe(var=var, obs=obs, selectvar=selectvar,
                                                    valuelabel=valuelabel, missingval=missingval)
        return data

    @staticmethod
    def save(path: str, frame: str | None = None,
             var: str | int | list = None,
             obs:  str | int | list = None,
             selectvar:  str | int | list = None,
             valuelabel: bool = False, missingval: str = _DefaultMissing(), *args, **kwargs):
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
            data = pystata.stata.frame_to_pdataframe(frame, var=var, obs=obs, selectvar=selectvar,
                                                     valuelabel=valuelabel, missingval=missingval)
        else:
            data = pystata.stata.data_to_pdataframe(var=var, obs=obs, selectvar=selectvar,
                                                    valuelabel=valuelabel, missingval=missingval)

        if extension == 'csv':
            data.to_csv(path, *args, **kwargs)
        elif extension == 'xlsx':
            data.to_excel(path, *args, **kwargs)
        elif extension == 'dta':
            data.to_stata(path, *args, **kwargs)
        elif extension == 'parquet':
            data.to_parquet(path, *args, **kwargs)
        elif extension == 'feather':
            data.to_feather(path, *args, **kwargs)
        elif extension == 'sas':
            data.to_sas(path, *args, **kwargs)
        elif extension == 'spss':
            data.to_spss(path, *args, **kwargs)
        elif extension == 'html':
            data.to_html(path, *args, **kwargs)
        elif extension == 'json':
            data.to_json(path, *args, **kwargs)
        elif extension in ['pkl', 'pickle', 'tar', 'gz', 'bz2', 'xz', 'zip']:
            data.to_pickle(path, *args, **kwargs)
        elif extension == 'sql':
            data.to_sql(path, *args, **kwargs)
        elif extension == 'clipboard':
            data.to_clipboard(*args, **kwargs)
        elif extension == 'xml':
            data.to_xml(path, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {extension}. Check "
                             f"https://pandas.pydata.org/docs/reference/io.html for supported file types.\n"
                             f"Is there a filetype you would like to see supported (pandas or not)? create a new issue "
                             f"on the Github repo or email zoellercollin@gmail.com.")

        return None

    @staticmethod
    def _parse_cmd(cmd: str, params: Dict):
        """
        parse elements of a StataHelper command and replace bracketed arguments with values from the arguments
        :param cmd: str StataHelper command
        :param params: dict of arguments
        :return: result of the command
        """
        for key, value in params.items():
            if isinstance(value, list):
                cmd = cmd.replace(f"{{{key}}}", " ".join(value))
            else:
                cmd = cmd.replace(f"{{{key}}}", value)
        return cmd

    def _prep_output(self):
        if self.output_dir is None and self.input_dir is not None:
            self.output_dir = os.path.join(self.input_dir, "output")
        elif self.output_dir is None and self.input_dir is None:
            self.output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(self.output_dir, exist_ok=True)

    # @carriage_print
    def schedule(self, cmd: str, pmap: dict, save_do=False):
        """
        Return the que of commands to be run in parallel (cartesian product). Analogous to the parallel method, but
        does not execute the commands.
        :param cmd: str StataHelper command template
        :param pmap: dict where the keys correspond with the values in cmd to change and the values are lists of values
        :param save_do: bool, save the commands to a do file in the output directory
        :return: list of commands to be run in parallel
        """

        # TODO: save schedule to a file and save to output_dir
        # Todo: option to save individual commands to do files in output_dir
        if not isinstance(cmd, str):
            raise TypeError(f" Invalid StataHelper command. Expected a string, got type {type(cmd)}.")

        # validate only keys in cmd are in pmap
        cmdkeys = literal_search(cmd)
        bad_pmap_keys = [k for k in pmap.keys() if k not in cmdkeys]  # keys in pmap but not in cmd
        bad_cmd_keys = [k for k in cmdkeys if k not in pmap.keys()]  # keys in cmd but not in pmap

        if bad_pmap_keys:
            bad_pmap_keys = '\n'.join(bad_pmap_keys)
            raise ValueError(f"The following key(s) are in pmap but not in cmd:\n"
                             f"     {bad_pmap_keys}")
        if bad_cmd_keys:
            bad_cmd_keys = '\n'.join(bad_cmd_keys)
            raise ValueError(f"The following key(s) are in cmd but not in pmap:\n"
                             f"     {bad_cmd_keys}")

        cartesian_args = cartesian(pmap.values())
        process_maps = [dict(zip(pmap.keys(), c)) for c in cartesian_args]
        self.queue = [self._parse_cmd(cmd, i) for i in process_maps]
        self.qcount = len(self.queue)
        self._prep_output()
        self.cmd = cmd
        self.pmap = pmap

        if save_do:
            with open(os.path.join(self.output_dir, "schedule.do"), "w") as f:
                f.write("\n".join(self.queue)) # fixme: needs to create a new dofile for each command, complete with the command, the output file, and logfile

        return self.queue

    def _parallel_task(self, idx, cmd, kwargs=None):

        name = self.savename + f"_{idx}"
        cmd = cmd.replace("*", name)

        fmt = "%d %b %Y %H:%M"
        starttime = time.time()
        print(f"{datetime.datetime.now().strftime(fmt)} :: Starting task {idx+1} of {self.qcount}")

        import pystata
        pystata.config.init(edition='mp', splash=True)
        pystata.stata.run(cmd, **kwargs)

        elapsed = time.time() - starttime
        print(f"{datetime.datetime.now().strftime(fmt)} ({elapsed:.4f}s) :: Finished task {idx+1} of {self.qcount}")
        return

    def parallel(self,
                 cmd: str,
                 pmap: dict,
                 name: str = None,
                 maxcores: int = None,
                 safety_buffer: int = 1,
                 **kwargs):
        """
        run a StataHelper command in parallel
        :param cmd: Template of Stata command
        :param pmap: dict where the keys correspond with the values in cmd to change and the values are lists of values
        :param name: str, base name of the output file to replace the wildcard '*'. None= "". Each process file is named
        according to its index in the queue, e.g. 'output_1', 'output_2', etc.
        :param maxcores: int, maximum number of cores to use. default is the number of cores on the machine.
        :param safety_buffer: int, number of cores to leave available. default is 1.
        """
        self.cmd = cmd
        self.schedule(cmd, pmap)
        self.savename = name if name is not None else ""

        # add pystata kwargs if they exist. Enumerate to get index in queue
        if kwargs:
            params = list(enumerate([(i, kwargs) for i in self.queue]))

            params = [(i, j[0], j[1]) for i, j in params]  # remove the interior tuple
        else:
            params = list(enumerate([i for i in self.queue]))
            params = [(i, j) for i, j in params]

        self.cores = limit_cores(params, maxcores, safety_buffer)
        print(f"\n# cmds in queue: {self.qcount}    # cores: {self.cores}\n")

        parallelize(func=self._parallel_task, iterable=params, cores=self.cores)

        return self
