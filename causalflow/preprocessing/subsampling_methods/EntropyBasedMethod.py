"""
This module provides the EntropyBasedMethod class.

Classes:
    EntropyBasedMethod: EntropyBasedMethod abstract class.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from causalflow.preprocessing.subsampling_methods.moving_window import MovingWindow

    
class EntropyBasedMethod(ABC):
    """EntropyBasedMethod abstract class."""
    
    def __init__(self, threshold):
        """
        Class constructor.

        Args:
            threshold (float): entropy threshold.
        """
        self.windows = list()
        self.segments = list()
        self.threshold = threshold


    def create_rounded_copy(self):
        """
        Create deepcopy of the dataframe but with rounded values.

        Returns:
            (pd.DataFrame): rounded dataframe.
        """
        de = deepcopy(self.df)
        de = de.round(1)
        return de


    def __normalization(self):
        """Normalize entropy for each moving window."""
        max_e = max([mw.entropy for mw in self.windows])
        for mw in self.windows:
            mw.entropy = mw.entropy / max_e


    def moving_window_analysis(self):
        """Compute dataframe entropy on moving windows."""
        de = self.create_rounded_copy()

        for ll, rl in self.segments:
            # Create moving window
            mw_df = de.values[ll: rl]

            # Build a Moving Window
            mw = MovingWindow(mw_df)

            # Compute entropy
            mw.get_entropy()

            # Compute optimal number of samples
            mw.optimal_sampling(self.threshold)

            # Collect result in a list
            self.windows.append(mw)

        # Entropy normalization
        self.__normalization()


    def extract_indexes(self):
        """Extract a list of indexes corresponding to the samples selected by the subsampling procedure."""
        _sample_index_list = list()
        for i, mw in enumerate(self.windows):
            sum_ws = sum([wind.T for wind in self.windows[:i]])
            sample_index = [si + sum_ws for si in mw.opt_samples_index]
            _sample_index_list += sample_index
        return _sample_index_list


    @abstractmethod
    def dataset_segmentation(self):
        """Abstract method."""
        pass