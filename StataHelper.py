"""
StataHelper: a simplified Python wrapper and parallelizer for StataHelper
"""
import sys
from builtins import *
from typing import List, Tuple, Dict
from utils import *
from utils import _DefaultMissing
from wrappers import *
import pandas as pd
import numpy as np
from collections import OrderedDict
import os
from glob import glob
import time
import datetime


class StataHelper:
    def __init__(self,
                 stata_path=None,
                 edition=None,
                 splash=None,
                 set_estimates_dir=None,
                 set_graph_format=None,
                 set_graph_size=None,
                 set_graph_show=None,
                 set_command_show=None,
                 set_autocompletion=None,
                 set_streaming_output=None,
                 set_output_file=None):

        # --------------------------- Module Parameters ---------------------------
        self.input_dir = None
        self.estimates_dir = set_estimates_dir
        self.overwrite_estimates = False
        self.cmd = None
        self.pmap = None
        self.queue = None
        self.qcount = None

        # --------------------------- System/ Parallelization Parameters ---------------------------
        self.cores = cpu_count()
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

        self._stata_initialized = pystata.config.is_stata_initialized()  # Doesn't seem to work in base Pystata module

        if not self.is_stata_initialized:
            raise SystemError("StataHelper is not initialized.")  # TODO: change to StataError

    def is_stata_initialized(self):
        """
        check if StataHelper is initialized: Wrapper for pystata.config.is_stata_initialized()
        """
        return self._stata_initialized

    @staticmethod
    def status():
        """
        check the status of the StataHelper instance. Wrapper for pystata.config.status()
        """
        import pystata
        return pystata.config.status()

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
        run a single StataHelper command. wrapper for pystata.stata.run()
        :param cmd: StataHelper command
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

    def use(self, dta: str, columns: List[str] | None = None, obs: str | None = None, **kwargs):
        """
        Inline method to use data in StataHelper
        :param dta: str dta path
        :param columns: list of columns to use when loading data
        :param obs: observations to load
        :return: None. Data is loaded into StataHelper
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

    @staticmethod
    def save(path: str, frame: str | None = None,
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

    def conditions(self, keys: List[str], sep="&", name="if"):
        """
        create a condition string for StataHelper
        :type sep: str
        :param keys: list of keys
        :param sep: separator
        :param name: name of the condition
        :return: condition string
        """
        subdict = {k: self.pmap[k] for k in keys}
        combinations = cartesian(subdict.values())
        conditions = [f"{sep}".join(map(str, c)) if "" not in c else " ".join(map(str, c)) for c in combinations]
        conditions = [f" if {c}" if c != "" else "" for c in conditions]
        self.pmap[name] = conditions
        self.pmap = {k: v for k, v in self.pmap.items() if k not in keys}
        return self

    @staticmethod
    def _parse_cmd(cmd: str, params: Dict):
        """
        parse elements of a StataHelper command and replace wildcards or bracketed arguments with values from the arguments
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

    def _prep_task_dict(self, items_dict, keys, names):
        subdict = OrderedDict({})  # preserve order
        cmd = self._parse_cmd(self.cmd, items_dict)
        idx = []
        for key, value in items_dict.items():
            if isinstance(self.pmap[key], str):
                idx.append(0)
            else:
                idx.append(self.pmap[key].index(value))
            if isinstance(value, list):
                if key is not None and key in keys:
                    if isinstance(names, list):
                        for i, item in enumerate(value):
                            subdict[item] = names[i]
                    else:
                        for i, item in enumerate(value):
                            subdict[item] = names
                else:
                    subdict[key] = sep(value)
            else:
                subdict[key] = value
        outname = "_".join(map(str, idx)) + ".ster"

        if self.estimates_dir is None and self.input_dir is not None:
            self.estimates_dir = os.path.join(self.input_dir, "estimates")
        elif self.estimates_dir is None and self.input_dir is None:
            self.estimates_dir = os.path.join(os.getcwd(), "estimates")
        os.makedirs(self.estimates_dir, exist_ok=True)

        if len(glob(os.path.join(self.estimates_dir, "*.ster"))) > 0 and self.overwrite_estimates is False:
            raise OverwriteError(self.estimates_dir, len(glob(os.path.join(self.estimates_dir, "*.ster"))))

        # if self.overwrite_estimates:
        #     for file in glob(os.path.join(self.estimates_dir, "*.ster")):
        #         os.remove(file)

        outname = os.path.join(self.estimates_dir, outname)

        subdict['outname'] = outname
        subdict['cmd'] = cmd
        return subdict

    # @carriage_print
    def schedule(self, cmd: str, pmap: dict):
        """
        Return the que of commands to be run in parallel (cartesian product). Analogous to the parallel method, but
        does not execute the commands.
        :param cmd: str StataHelper command template
        :param pmap: dict where the keys correspond with the values in cmd to change and the values are lists of values
        :return: list of commands to be run in parallel
        """

        if not isinstance(cmd, str):
            raise TypeError(f" Invalid StataHelper command. Expected a string, got type {type(cmd)}.")
        acutal_args = cmd.count("{")
        map_args = len(pmap.keys())
        if acutal_args > map_args:
            raise ValueError(f"Expected {map_args} parameters, but received {acutal_args}.")

        self.cmd = cmd
        cartesian_args = cartesian(pmap.values())
        itemized = [dict(zip(pmap.keys(), c)) for c in cartesian_args]

        self.queue = [self._parse_cmd(cmd, i) for i in itemized]
        self.qcount = len(self.queue)
        return self.queue

    @staticmethod
    def _parallel_task(self, paramsdict: Dict, *kwargs):

        opt= "replace" if self.overwrite_estimates else "append"

        kwargs = kwargs[0]
        fmt = "%d %b %Y %H:%M"
        starttime = time.time()
        print(f"{datetime.datetime.now().strftime(fmt)} :: {paramsdict['cmd']}")

        xtra_cmd = ""

        for key, value in paramsdict.items():
            if key not in ['cmd', 'outname']:
                xtra_cmd += f'estadd local {key} "{value}"\n'
        xtra_cmd += "qui: sum `e(depvar)' if e(sample)\nqui: estadd scalar Mean =r(mean)\n"
        import pystata
        pystata.stata.run(paramsdict['cmd'], **kwargs)
        pystata.stata.run(xtra_cmd, **kwargs)
        pystata.stata.run(f"estimates save {paramsdict['outname']}, {opt}", **kwargs)

        endtime = time.time()
        elasped = endtime - starttime
        print(f"{datetime.datetime.now().strftime(fmt)} ({elasped:.4f}s) :: {paramsdict['cmd']}")
        return

    def parallel(self,
                 cmd: str,
                 pmap: dict,
                 overwrite_estimates=False,
                 maxcores: int = None,
                 safety_buffer: int = 1,
                 **kwargs):
        """
        run a StataHelper command in parallel: wrapper for pystata.stata.Run() on multiple cores
        :param cmd: Template of StataHelper command
        :param pmap: dict where the keys correspond with the values in cmd to change and the values are lists of values
        :param overwrite_estimates: bool, if True, overwrite existing estimates files. default is False.
        :param maxcores: int, maximum number of cores to use. default is the number of cores on the machine.
        :param safety_buffer: int, number of cores to leave available. default is 1.
        """
        self.cmd = cmd
        self.schedule(cmd, pmap)
        self.overwrite_estimates = overwrite_estimates
        if kwargs:
            params = [(i, kwargs) for i in self.queue]
        else:
            params = [i for i in self.queue]
        self.cores = limit_cores(params, maxcores, safety_buffer)
        # TODO: add prep_task_dict to parallelize
        print(f"\n# cmds in queue: {self.qcount}    # cores: {self.cores}\n")

        parallelize(func=self._parallel_task, iterable=params, maxcores=self.cores, buffer=safety_buffer)

        return self

    @staticmethod
    def results(src,  labelsdict, dst=None, argstring=None, fmt=None, title=None, keep_estimate_files=True, *args,
                **kwargs):
        """
        create an excel file with the results of the parallelized StataHelper commands. This organizes each sheet by the
        first key in the labelsdict in separate sheets.
        :param src: source directory of the estimates files
        :param dst: destination & name of the Excel file. default is cwd with the name 'results{currentdate}.xlsx'
        :param labelsdict: dictionary of labels for the keys in the estimates files.
        :param argstring: cmd for estout if different from the default. Allows user to defile how estout will handle
        the results.
        :param fmt: (stata) string list of stata format for values in labels. Passed to estout as 'fmt()'
        :param title: (stata) title of the output. Passed to estout as 'title()'
        :param keep_estimate_files: bool, if True, keep the estimates files after creating the excel file.
         If False, delete the estimates files.
        :param args: additional arguments for stata.run()
        :param kwargs: additional keyword arguments for stata.run()
        :return: None
        """
        print("\nSaving results to Excel file...\n")
        files = sorted(glob(os.path.join(src, "*.ster")))
        label_keys = labelsdict.keys()
        labelsdict = OrderedDict(labelsdict)

        # --------------------------- Parameters in Parameters ---------------------------

params = {'y': ['mpg'], 'x': [['weight', 'length'], ['weight']]}
statapath = r"C:\Program Files\Stata18\utilities"
cmd = "regress {y} {x}"
s = StataHelper(stata_path=statapath, edition='mp', splash=True)
s.parallel(cmd, params)


