from enum import Enum

class CPLevel(Enum):
    NONE = 0
    WARN = 1
    INFO = 2
    DEBUG = 3


class CPrinter():
    def __init__(self):
        self.verbosity = None
    
    def set_verbosity(self, verbosity: CPLevel):
        """
        set verbosity level

        Args:
            verbosity (CPLevel): verbosity level
        """
        self.verbosity = verbosity
        
    def warning(self, msg: str):
        """
        write message iff verbosity >= WARN

        Args:
            msg (str): massage to write
        """
        if self.verbosity.value >= CPLevel.WARN.value: print(msg)

    def info(self, msg: str):
        """
        write message iff verbosity >= INFO

        Args:
            msg (str): massage to write
        """
        if self.verbosity.value >= CPLevel.INFO.value: print(msg)

    def debug(self, msg: str):
        """
        write message iff verbosity >= DEBUG

        Args:
            msg (str): massage to write
        """
        if self.verbosity.value >= CPLevel.DEBUG.value: print(msg)

CP = CPrinter()