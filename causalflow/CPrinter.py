"""
This module provides various classes for the creation of log during the causal discovery analysis.

Classes:
    CPLevel: support class for handling different levels of log.
    CPrinter: class responsible for the log.
"""

from enum import Enum

class CPLevel(Enum):
    """CPLevel Enumerator."""
    
    NONE = 0
    WARN = 1
    INFO = 2
    DEBUG = 3


class CPrinter():
    """CPrinter class."""
    
    def __init__(self):
        """Class constructor."""
        self.verbosity = None
    
    def set_verbosity(self, verbosity: CPLevel):
        """
        Set verbosity level.

        Args:
            verbosity (CPLevel): verbosity level.
        """
        self.verbosity = verbosity
        
    def warning(self, msg: str):
        """
        Write message iff verbosity >= WARN.

        Args:
            msg (str): massage to write.
        """
        if self.verbosity.value >= CPLevel.WARN.value: print(msg)

    def info(self, msg: str):
        """
        Write message iff verbosity >= INFO.

        Args:
            msg (str): massage to write.
        """
        if self.verbosity.value >= CPLevel.INFO.value: print(msg)

    def debug(self, msg: str):
        """
        Write message iff verbosity >= DEBUG.

        Args:
            msg (str): massage to write.
        """
        if self.verbosity.value >= CPLevel.DEBUG.value: print(msg)

CP = CPrinter()