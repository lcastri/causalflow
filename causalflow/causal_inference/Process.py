import numpy as np

class Process():
    def __init__(self, data: np.ndarray, varname, lag, nsample):
        """
        Process contructer

        Args:
            data (np.ndarray): process data
            varname (str): process name (e.g., X_0)
            lag (int): process lag time
            nsample (int): number of samples for density estimation
        """
        self.data = data.reshape(-1, 1)
        self.varname = varname
        self.pvarname = '$' + varname + '$'
        self.lag = lag
        self.nsample = nsample
        
    @property
    def T(self):
        """
        Returns data length

        Returns:
            int: data length
        """
        return len(self.data)
    
    @property
    def alignT(self):
        """
        Returns data length after alignment

        Returns:
            int: data length after alignment
        """
        return len(self.aligndata)
    
    def align(self, maxlag: int):
        """
        Aligns data w.r.t. maxLag between processes

        Args:
            maxlag (int): max time lag between processes

        Returns:
            ndarray: aligned data
        """
        self.aligndata = self.data[maxlag - self.lag : self.T - self.lag]
        return self.aligndata
    
    @property
    def samples(self):
        """
        Returns a ndarray of _nsample_ samples extracted from the original dataset 

        Returns:
            ndarray: _nsample_ samples of the original dataset
        """
        return np.squeeze(np.linspace(min(self.aligndata), max(self.aligndata), self.nsample))