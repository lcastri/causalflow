"""
This module provides the Logger class.

Classes:
    Logger: class responsible for the log.
"""

import sys
from causalflow.basics.utils import cls


class Logger(object):
    """Logger class."""
    
    def __init__(self, path, clean_console = True):
        """
        Class constructor.

        Args:
            path (str): log file path.
            clean_console (bool, optional): clean console flag. Defaults to True.
        """
        self.terminal = sys.stdout
        self.log = open(path, "w")
        if clean_console: cls()

    def write(self, message):
        """
        Write message.

        Args:
            message (str): log msg.
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """python3 compatibility."""
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
    
    def close(self):
        """Close logger."""
        sys.stdout = sys.__stdout__
        self.log.close()