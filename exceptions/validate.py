"""
Handles all the exceptions that are raised when validating the input data.
"""


class _DefaultMissing:
    def __repr__(self):
        return "_DefaultMissing()"


class PyataException(Exception):
    """
    Base class for all exceptions in Stata
    """
    pass

