import numpy as np
from causalflow.basics.constants import *

class Process():
    def __init__(self, 
                 data: np.ndarray, 
                 varname: str, 
                 lag: int, 
                 data_type: DataType, 
                 node_type: NodeType):
        """
        Process contructer

        Args:
            data (np.ndarray): process data
            varname (str): process name (e.g., X_0)
            lag (int): process lag time
            data_type (DataType): data type (continuous|discrete)
        """
        self.data = data.reshape(-1, 1)
        self.data = np.array([x[0] for x in self.data], dtype=np.float32)
        self.varname = varname
        self.pvarname = '$' + varname + '$'
        self.lag = lag
        self.data_type = data_type
        self.node_type = node_type
        
        self.aligndata = None
        
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
            self.aligndata = np.array(self.data[maxlag - self.lag : self.T - self.lag], dtype=np.float32).reshape(-1, 1)
            return self.aligndata
        else:
            return None