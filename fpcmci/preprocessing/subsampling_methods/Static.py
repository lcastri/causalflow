from fpcmci.preprocessing.subsampling_methods.SubsamplingMethod import SubsamplingMethod, SSMode


class Static(SubsamplingMethod):
    """
    Subsamples data by taking one sample each step-samples
    """
    def __init__(self, step):
        """
        Static class constructor
        
        Args:
            step (int): integer subsampling step

        Raises:
            ValueError: if step == None
        """
        super().__init__(SSMode.Static)
        if step is None:
            raise ValueError("step not specified")
        self.step = step

    def run(self):
        return range(0, len(self.df.values), self.step)