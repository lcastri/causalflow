from numpy import ndarray

class Process():
    def __init__(self, data: ndarray, varname, lag):
        self.data = data.reshape(-1, 1)
        self.varname = varname
        self.pvarname = '$' + varname + '$'
        self.lag = lag
        self.T = len(data)
    
    def align(self, maxlag: int):
        return self.data[maxlag - self.lag : self.T - self.lag]