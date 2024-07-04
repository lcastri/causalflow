import sys
from causalflow.basics.utils import cls


class Logger(object):
    def __init__(self, path, clean_console = True):
        self.terminal = sys.stdout
        self.log = open(path, "w")
        if clean_console: cls()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
    
    def close(self):
        sys.stdout = sys.__stdout__
        self.log.close()