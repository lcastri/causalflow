import numpy as np
from causalflow.basics.constants import *
import numpy as np

class Process():
    def __init__(self, data: np.ndarray, varname: str, lag: int, nsample: int, data_type: DataType, node_type: NodeType):
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
        # self.nsample = self.estimate_optimal_samples()
        
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
            self.original_indices = self.get_original_indices()
            return self.aligndata
        else:
            return None
        
    
    # def get_samples(self):
    #     """
    #     Returns a ndarray of _nsample_ samples extracted from the original dataset 

    #     Returns:
    #         ndarray: _nsample_ samples of the original dataset
    #     """
    #     if self.node_type is not NodeType.Context:
    #         return np.squeeze(np.linspace(min(self.aligndata), max(self.aligndata), self.nsample, dtype = int if self.data_type is DataType.Discrete else None))
    #     else:
    #         return np.unique(self.data)


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
            
            # Sort aligndata for even sampling
            # Generate evenly spaced indices within the range of sorted data
            sorted_data = np.sort(self.aligndata, axis=0)
            self.sorted_samples = np.array(sorted_data[indices], dtype=np.float32 if self.data_type is DataType.Continuous else np.int16)

        else:
            self.sorted_samples = np.unique(self.data)  # Handle 'Context' node type
            self.samples = np.unique(self.data)  # Handle 'Context' node type
        
        
    def get_original_indices(self):
        original_indices = []
        for v in self.aligndata:
            if v in list(self.sorted_samples) and list(self.sorted_samples).index(v) not in original_indices:
                original_indices.append(list(self.sorted_samples).index(v))
        return original_indices
    
    @property
    def original_samples(self):
        return self.sorted_samples[self.original_indices]