"""
This package is a collection of utilities and wrappers for Stata.
The utils.py file is a collection of utility functions that are used by the wrappers.
The wrappers.py file is a collection of classes that are used to wrap Stata commands and functions.
The StataHelper.py file is the main class that is used to interface with Stata.
The sfi-tools.py file is a collection of tools that are used to interface with Stata (future update).

The author makes no claims to the accuracy or reliability of the code as it relates to the Stata software. Although
comprehensive testing has been performed, use of this package is on an as-is basis and the author or contributors are
not responsible for any damages or losses that may occur from incorrect use or tampering of the software.
Notwithstanding, users are encouraged to report any issues or bugs, as well as fork the repository and make changes as
they see fit. See the GitHub repository for more information on contributing or reporting issues.

The author is not affiliated with StataCorp, nor is this package endorsed by StataCorp. Stata is a registered trademark
of StataCorp LLC.
"""


from .StataHelper import StataHelper
from . import utils
from . import wrappers
version = "0.1.0"
__version__ = version
__author__ = "Collin Zoeller"

__all__ = ['StataHelper', 'utils', 'wrappers']
