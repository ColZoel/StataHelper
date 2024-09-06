"""
This package is a collection of utilities and wrappers for Stata.
The utils.py file is a collection of utility functions that are used by the wrappers.
The wrappers.py file is a collection of classes that are used to wrap Stata commands and functions.
The StataHelper.py file is the main class that is used to interface with Stata.
The sfi-tools.py file is a collection of tools that are used to interface with Stata (future update).
"""


from .StataHelper import StataHelper
from . import utils
from . import wrappers
version = "1.0.1"
__version__ = version
__author__ = "Collin Zoeller"

__all__ = ['StataHelper', 'utils', 'wrappers', '__version__', '__author__']
