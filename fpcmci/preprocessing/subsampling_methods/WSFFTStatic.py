from scipy.fft import rfft, rfftfreq
from fpcmci.preprocessing.subsampling_methods.EntropyBasedMethod import EntropyBasedMethod
from fpcmci.preprocessing.subsampling_methods.SubsamplingMethod import SSMode, SubsamplingMethod
import numpy as np
import pylab as pl
from math import ceil
import scipy.signal


class WSFFTStatic(SubsamplingMethod, EntropyBasedMethod):
    """
    Subsampling method with static window size based on Fourier analysis
    """
    def __init__(self, sampling_time, entropy_threshold):
        """
        WSFFTStatic class constructor

        Args:
            sampling_time (float): timeseries sampling time
            entropy_threshold (float): entropy threshold
        """
        SubsamplingMethod.__init__(self, SSMode.WSFFTStatic)
        EntropyBasedMethod.__init__(self, entropy_threshold)
        self.sampling_time = sampling_time


    def __fourier_window(self):
        """
        Compute window size based on Fourier analysis performed on dataframe

        Returns:
            (int): window size
        """
        N, dim = self.df.shape
        xf = rfftfreq(N, self.sampling_time)
        w_array = list()
        for i in range(0, dim):
            yf = np.abs(rfft(self.df.values[:, i]))

            peak_indices, _ = scipy.signal.find_peaks(yf)
            highest_peak_index = peak_indices[np.argmax(yf[peak_indices])]
            w_array.append(ceil(1 / (2 * xf[highest_peak_index]) / self.sampling_time))
            # fig, ax = pl.subplots()
            # ax.plot(xf, yf)
            # ax.plot(xf[highest_peak_index], np.abs(yf[highest_peak_index]), "x")
            # pl.show()
        return min(w_array)


    def dataset_segmentation(self):
        """
        Segments dataset with a fixed window size
        """
        seg_res = [i for i in range(0, len(self.df.values), self.ws)]
        self.segments = [(i, i + self.ws) for i in range(0, len(self.df.values) - self.ws, self.ws)]
        if not seg_res.__contains__(len(self.df.values)):
            self.segments.append((seg_res[-1], len(self.df.values)))
            seg_res.append(len(self.df.values))

    
    def run(self):
        """
        Run subsampler

        Returns:
            (list[int]): indexes of the remaining samples
        """
        # define window size
        self.ws = self.__fourier_window()

        # build list of segment
        self.dataset_segmentation()

        # compute entropy moving window
        self.moving_window_analysis()

        # extracting subsampling procedure results
        idxs = self.extract_indexes()

        return idxs