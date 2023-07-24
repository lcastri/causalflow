from abc import ABC, abstractmethod
from copy import deepcopy
from fpcmci.preprocessing.subsampling_methods.moving_window import MovingWindow

    
class EntropyBasedMethod(ABC):
    """
    EntropyBasedMethod abstract class
    """
    def __init__(self, threshold):
        self.windows = list()
        self.segments = list()
        self.threshold = threshold


    def create_rounded_copy(self):
        """
        Create deepcopy of the dataframe but with rounded values

        Returns:
            (pd.DataFrame): rounded dataframe
        """
        de = deepcopy(self.df)
        de = de.round(1)
        return de


    def __normalization(self):
        """
        Normalize entropy for each moving window
        """
        max_e = max([mw.entropy for mw in self.windows])
        for mw in self.windows:
            mw.entropy = mw.entropy / max_e


    def moving_window_analysis(self):
        """
        Compute dataframe entropy on moving windows
        """
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


    # def extract_data(self):
    #     """
    #     Extract plottable data from moving window analysis
    #     """
    #     # Entropies and samples numbers list
    #     self.__entropy_list = [mw.entropy for mw in self.__window_list]
    #     self.__sample_number_list = [mw.opt_size for mw in self.__window_list]
    #     self.__original_size = [mw.T for mw in self.__window_list]
    #     self.num_samples = sum(self.__sample_number_list)

    #     # Make entropy and sample array plottable
    #     self.__pretty_signals()


    # def __pretty_signals(self):
    #     """
    #     Make entropy list and sample number list plottable
    #     """
    #     _pretty_entropy = []
    #     _pretty_sample_number = []
    #     _pretty_original_size = []
    #     for i, mw in enumerate(self.__window_list):
    #         _pretty_entropy += np.repeat(self.__entropy_list[i], mw.T).tolist()
    #         _pretty_sample_number += np.repeat(self.__sample_number_list[i], mw.T).tolist()
    #         _pretty_original_size += np.repeat(self.__original_size[i], mw.T).tolist()
    #     self.__entropy_list = _pretty_entropy
    #     self.__sample_number_list = _pretty_sample_number
    #     self.__original_size = _pretty_original_size

    #     _diff = self.df.shape[0] - len(self.__entropy_list)
    #     if _diff != 0:
    #         self.__entropy_list = np.append(self.__entropy_list, [self.__entropy_list[-1]] * _diff)
    #         self.__sample_number_list = np.append(self.__sample_number_list, [self.__sample_number_list[-1]] * _diff)


    def extract_indexes(self):
        """
        Extract a list of indexes corresponding to the samples
        selected by the subsampling procedure
        """
        _sample_index_list = list()
        for i, mw in enumerate(self.windows):
            sum_ws = sum([wind.T for wind in self.windows[:i]])
            sample_index = [si + sum_ws for si in mw.opt_samples_index]
            _sample_index_list += sample_index
        return _sample_index_list


    @abstractmethod
    def dataset_segmentation(self):
        """
        abstract method
        """
        pass