"""
This module provides the Static class.

Classes:
    Static: Subsamples data by taking one sample each step-samples.
"""

from causalflow.preprocessing.subsampling_methods.SubsamplingMethod import SubsamplingMethod, SSMode

class Static(SubsamplingMethod):
    """Subsample data by taking one sample each step-samples."""
    
    def __init__(self, step):
        """
        Class constructor.
        
        Args:
            step (int): integer subsampling step.

        Raises:
            ValueError: if step == None.
        """
        super().__init__(SSMode.Static)
        if step is None:
            raise ValueError("step not specified")
        self.step = step

    def run(self):
        """Run subsampler."""
        return range(0, len(self.df.values), self.step)