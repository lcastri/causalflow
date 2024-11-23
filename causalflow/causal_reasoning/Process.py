import numpy as np
from causalflow.basics.constants import *
import numpy as np

class Process():
    def __init__(self, 
                 data: np.ndarray, 
                 varname: str, 
                 lag: int, 
                 nsample: int, 
                 data_type: DataType, 
                 node_type: NodeType):
        """
        Process contructer

        Args:
            data (np.ndarray): process data
            varname (str): process name (e.g., X_0)
            lag (int): process lag time
            nsample (int): number of samples for density estimation
            data_type (DataType): data type (continuous|discrete)
        """
        self.data = data.reshape(-1, 1)
        self.data = np.array([x[0] for x in self.data], dtype=np.float32)
        self.varname = varname
        self.pvarname = '$' + varname + '$'
        self.lag = lag
        self.nsample = nsample
        self.data_type = data_type
        self.node_type = node_type
        
        self.aligndata = None
        self.samples = None
        self.sorted_samples = None
        self.original_indices = None
        
    @property
    def T(self):
        """
        Returns data length

        Returns:
            int: data length
        """
        return len(self.data) if self.node_type is not NodeType.Context else len(np.unique(self.data))
    
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
        if self.node_type is not NodeType.Context:
            self.aligndata = np.array(self.data[maxlag - self.lag : self.T - self.lag], dtype=np.float32)
            self.get_samples()
            # self.original_indices = self.get_original_indices()
            return self.aligndata
        else:
            return None
        
    
    def get_samples(self):
        """
        Returns an ndarray of _nsample_ samples extracted from the original dataset,
        with samples spaced approximately evenly from min to max within `aligndata`.

        Returns:
            ndarray: _nsample_ samples of the original dataset
        """
        if self.node_type is not NodeType.Context:            
            indices = np.linspace(0, len(self.aligndata) - 1, self.nsample, dtype=int)
            self.samples = np.array(self.aligndata[indices], dtype=np.float32 if self.data_type is DataType.Continuous else np.int16)
        else:
            self.samples = np.unique(self.data)