"""
Tools for SFI interfacing. These are used within StataHelper to better interface with Python.
"""
import numpy.typing as npt
import numpy as np
import pandas as pd


def use(path: npt.ArrayLike | str, frame: None | str = None, force=False, *args, **kwargs):
    """
    read any pandas supported file type and send to StataHelper instance
    """
    if isinstance(path, np.ndarray):
        if frame is not None:
            pystata.stata.nparray_to_frame(path, frame, force=force)
        else:
            pystata.stata.nparray_to_data(path, force=force)
        return

    elif isinstance(path, pd.DataFrame) or isinstance(path, pd.Series):
        data = path
    elif isinstance(path, str):
        extension = path.split('.')[-1]
        if extension == 'csv':
            data = pd.read_csv(path, *args, **kwargs)
        elif extension == 'xlsx':
            data = pd.read_excel(path, *args, **kwargs)
        elif extension == 'dta':
            data = pd.read_stata(path, *args, **kwargs)
        elif extension == 'parquet':
            data = pd.read_parquet(path, *args, **kwargs)
        elif extension == 'feather':
            data = pd.read_feather(path, *args, **kwargs)
        elif extension == 'sas':
            data = pd.read_sas(path, *args, **kwargs)
        elif extension == 'spss':
            data = pd.read_spss(path, *args, **kwargs)
        elif extension == 'html':
            data = pd.read_html(path, *args, **kwargs)
        elif extension == 'json':
            data = pd.read_json(path, *args, **kwargs)
        elif extension in ['pkl', 'pickle', 'tar', 'gz', 'bz2', 'xz', 'zip']:
            data = pd.read_pickle(path, *args, **kwargs)
        elif extension == 'sql':
            data = pd.read_sql(path, *args, **kwargs)
        elif extension == 'clipboard':
            data = pd.read_clipboard(*args, **kwargs)
        elif extension == 'xml':
            data = pd.read_xml(path, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {extension}. Check "
                             f"https://pandas.pydata.org/docs/reference/io.html for supported file types.\n"
                             f"Is your filetype supported by pandas but not listed here? Email zoellercollin@gmail.com"
                             f"or open an issue on the Github repo to make it right.")
    else:
        raise ValueError("Unsupported file type. Array, Pandas objects, or saved files accepted. Check "
                         "https://pandas.pydata.org/docs/reference/io.html for supported file types.\n")

    if frame is not None:
        pystata.stata.pdataframe_to_frame(data, frame, force=force)
    else:
        pystata.stata.pdataframe_to_data(data, force=force)

    return
