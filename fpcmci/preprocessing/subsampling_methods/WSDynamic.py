import ruptures as rpt
from fpcmci.preprocessing.subsampling_methods.EntropyBasedMethod import EntropyBasedMethod
from fpcmci.preprocessing.subsampling_methods.SubsamplingMethod import SubsamplingMethod, SSMode


class WSDynamic(SubsamplingMethod, EntropyBasedMethod):
    """
    Subsampling method with dynamic window size based on entropy analysis
    """
    def __init__(self, window_min_size, entropy_threshold):
        """
        WSDynamic class constructor

        Args:
            window_min_size (int): minimun window size
            entropy_threshold (float): entropy threshold

        Raises:
            ValueError: if window_min_size == None
        """
        SubsamplingMethod.__init__(self, SSMode.WSDynamic)
        EntropyBasedMethod.__init__(self, entropy_threshold)
        if window_min_size is None:
            raise ValueError("window_type = DYNAMIC but window_min_size not specified")
        self.wms = window_min_size
        self.ws = None

    def dataset_segmentation(self):
        """
        Segments dataset based on breakpoint analysis and a min window size
        """
        de = self.create_rounded_copy()
        algo = rpt.Pelt(model = "l2", min_size = self.wms).fit(de)
        seg_res = algo.predict(pen = 10)
        self.segments = [(seg_res[i - 1], seg_res[i]) for i in range(1, len(seg_res))]
        self.segments.insert(0, (0, seg_res[0]))


    def run(self):
        """
        Run subsampler

        Returns:
            (list[int]): indexes of the remaining samples
        """
        # build list of segment
        self.dataset_segmentation()

        # compute entropy moving window
        self.moving_window_analysis()

        # extracting subsampling procedure results
        idxs = self.extract_indexes()

        return idxs