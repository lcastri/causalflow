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
        """
        Class constructor.
        """
        self.verbosity = CPLevel.NONE
        self.log_file = None
        
    def set_verbosity(self, verbosity: CPLevel):
        """
        Set verbosity level.

        Args:
            verbosity (CPLevel): verbosity level.
        """
        self.verbosity = verbosity
        
    def set_logpath(self, log_file: str = "log.txt"):
        """
        Set verbosity level.

        Args:
            verbosity (CPLevel): verbosity level.
        """
        self.log_file = log_file

    def _log(self, level: str, msg: str, noTXT: bool, noConsole: bool):
        """
        Log a message to both the console and the log file.

        Args:
            msg (str): Message to log.
        """
        if msg.startswith('\n'):
            msg = f"\n{level}: {msg[1:]}"
        else:
            msg = f"{level}: {msg}"
        if not noConsole: print(msg)  # Print to console
        if not noTXT: 
            with open(self.log_file, 'a') as file:
                file.write(msg + "\n")  # Append to log file

    def warning(self, msg: str, noTXT = False, noConsole = False):
        """
        Write message iff verbosity >= WARN.

        Args:
            msg (str): Message to write.
        """
        if self.verbosity.value >= CPLevel.WARN.value:
            if self.log_file is not None:
                self._log("[WARNING]", msg, noTXT, noConsole)
            else:
                print(msg)

    def info(self, msg: str, noTXT = False, noConsole = False):
        """
        Write message iff verbosity >= INFO.

        Args:
            msg (str): Message to write.
        """
        if self.verbosity.value >= CPLevel.INFO.value:
            if self.log_file is not None:
                self._log("[INFO]", msg, noTXT, noConsole)
            else:
                print(msg)

    def debug(self, msg: str, noTXT = False, noConsole = False):
        """
        Write message iff verbosity >= DEBUG.

        Args:
            msg (str): Message to write.
        """
        if self.verbosity.value >= CPLevel.DEBUG.value:
            if self.log_file is not None:
                self._log("[DEBUG]", msg, noTXT, noConsole)
            else:
                print(msg)
CP = CPrinter()