from tigramite.independence_tests.gpdc import GPDC
from ts_causal_discovery.CPrinter import CPLevel
from ts_causal_discovery.FPCMCI import FPCMCI
from ts_causal_discovery.preprocessing.data import Data
from ts_causal_discovery.preprocessing.subsampling_methods.Static import Static
from ts_causal_discovery.preprocessing.subsampling_methods.SubsamplingMethod import SubsamplingMethod
from ts_causal_discovery.preprocessing.subsampling_methods.WSDynamic import WSDynamic
from ts_causal_discovery.preprocessing.subsampling_methods.WSFFTStatic import WSFFTStatic
from ts_causal_discovery.preprocessing.subsampling_methods.WSStatic import WSStatic
from ts_causal_discovery.selection_methods.TE import TE, TEestimator
from ts_causal_discovery.basics.constants import LabelType
import numpy as np

from time import time
from datetime import timedelta


if __name__ == '__main__':   
    alpha = 0.05
    min_lag = 1
    max_lag = 1
    
    np.random.seed(1)
    nsample = 500
    nfeature = 6
    d = np.random.random(size = (nsample, nfeature))
    for t in range(max_lag, nsample):
        d[t, 0] += 2 * d[t-1, 1] + 3 * d[t-1, 3]
        d[t, 2] += 1.1 * d[t-1, 1]**2
        d[t, 3] += d[t-1, 3] * d[t-1, 2]
        d[t, 4] += d[t-1, 4] + d[t-1, 5] * d[t-1, 0]

    df = Data(d)
    start = time()
    FS = FPCMCI(df, 
                f_alpha = alpha, 
                pcmci_alpha = alpha, 
                min_lag = min_lag, 
                max_lag = max_lag, 
                sel_method = TE(TEestimator.Gaussian), 
                val_condtest = GPDC(significance = 'analytic', gp_params = None),
                verbosity = CPLevel.DEBUG,
                neglect_only_autodep = False,
                resfolder = 'ex_PCMCI')
    
    selector_res = FS.run_pcmci()
    print(FS.f_deps)
    elapsed_PCMCI = time() - start
    print(str(timedelta(seconds = elapsed_PCMCI)))
    FS.dag(label_type = LabelType.NoLabels, node_layout = 'circular')


    
    

