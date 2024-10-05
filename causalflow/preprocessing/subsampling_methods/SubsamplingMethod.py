"""
This module provides subsampling methods for data preprocessing.

Classes:
    SSMode: An enumerator containing all the supported subsampling methods.
    SubsamplingMethod: A class for implementing various subsampling techniques.
"""

from abc import ABC, abstractmethod
from enum import Enum
import pandas as pd


class SSMode(Enum):
    """Enumerator containing all the supported subsampling methods."""

    WSDynamic = 'Dynamic-size moving window'
    WSStatic = 'Static-size moving window'
    WSFFTStatic = 'FFT static-size moving window'
    Static = 'Static'
    

class SubsamplingMethod(ABC):
    """SubsamplingMethod abstract class."""
    
    def __init__(self, ssmode: SSMode):
        """
        Class constructor.

        Args:
            ssmode (SSMore): Subsampling method.
        """
        self.ssmode = ssmode
        self.df = None

    
    def initialise(self, dataframe: pd.DataFrame):
        """
        Initialise class by setting the dataframe to subsample.

        Args:
            dataframe (pd.DataFrame): Pandas DataFrame to subsample.
        """
        self.df = dataframe


    @abstractmethod
    def run(self):
        """Run subsampler."""
        pass
