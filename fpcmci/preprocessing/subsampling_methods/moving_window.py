import math
from scipy.stats import entropy


class MovingWindow:
    def __init__(self, window):
        self.window = window
        self.T, self.dim = window.shape
        self.entropy = None
        self.opt_size = None
        self.opt_samples_index = None

    def get_pdf(self):
        """
        Compute the probability distribution function from an array of data

        Returns:
            list: probability distribution function
        """

        counts = {}

        for i in range(0, self.T):
            t = tuple(self.window[i, :])
            if t in counts:
                counts[t] += 1
            else:
                counts[t] = 1

        pdf = {k: v / self.T for k, v in counts.items()}

        return list(pdf.values())


    def get_entropy(self):
        """
        Compute the entropy based on probability distribution function
        """
        self.entropy = entropy(self.get_pdf(), base = 2)


    def samples_selector(self, step):
        """
        Select sample to be taken from a moving window


        Args:
            step (int): subsampling frequency

        Returns:
            list[int]: list of indexes corresponding to the sample to be taken
        """
        return [i for i in range(0, self.T, step)]


    def optimal_sampling(self, thres):
        """
        Find the optimal number of sample for a particular moving window


        Args:
            thres (float): stopping criteria threshold
        """
        converged = False
        _old_step = 0
        _sub_index = list(range(0, self.T))
        _old_sub_index = list(range(0, self.T))
        _max_n = math.floor(self.T / 2)

        for n in range(_max_n, 1, -1):
            # resampling window with n samples and build another Moving Window
            step = int(self.T / n)
            if step == _old_step:
                continue
            _old_step = step
            _old_sub_index = _sub_index
            _sub_index = self.samples_selector(step)
            _sub_w = MovingWindow(self.window[_sub_index])

            # compute entropy on the sub window
            _sub_w.get_entropy()

            # stopping criteria
            if self.entropy != 0:
                if abs(_sub_w.entropy - self.entropy) / self.entropy >= thres:
                    converged = True
                    break
        self.opt_size = len(_old_sub_index) if converged else len(_sub_index)
        self.opt_samples_index = _old_sub_index if converged else _sub_index
