import numpy as np

class Process():
    def __init__(self, data: np.ndarray, varname, lag, nsample):
        self.data = data.reshape(-1, 1)
        self.varname = varname
        self.pvarname = '$' + varname + '$'
        self.lag = lag
        self.nsample = nsample
        
    @property
    def T(self):
        return len(self.data)
    
    @property
    def alignT(self):
        return len(self.aligndata)
    
    def align(self, maxlag: int):
        self.aligndata = self.data[maxlag - self.lag : self.T - self.lag]
        return self.aligndata
    
    @property
    def samples(self):
        return np.linspace(min(self.aligndata), max(self.aligndata), self.nsample)