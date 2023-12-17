from tigramite.independence_tests.gpdc import GPDC
from connectingdots.CPrinter import CPLevel
from connectingdots.causal_discovery.FPCMCI import FPCMCI
from connectingdots.causal_discovery.baseline.DYNOTEARS import DYNOTEARS
from connectingdots.preprocessing.data import Data
from connectingdots.selection_methods.TE import TE, TEestimator
from connectingdots.basics.constants import LabelType
import numpy as np


if __name__ == '__main__':   
    f_alpha = 0.1
    pcmci_alpha = 0.05
    min_lag = 1
    max_lag = 2

    np.random.seed(1)
    nsample = 1000
    nfeature = 5
    d = np.random.random(size = (nsample, nfeature))
    for t in range(max_lag, nsample):
        d[t, 1] += 0.5 * d[t-1, 0]**2
        d[t, 2] += 0.3 * d[t-1, 0] * 0.75 * d[t-2, 1] - 2.5
        d[t, 3] += 0.7 * d[t-1, 3] * d[t-2, 4]
       
    df = Data(d, vars = ['X_0', 'X_1', 'X_2', 'X_3', 'X_4'])
    
    
    fpcmci = FPCMCI(df,
                    f_alpha = f_alpha, 
                    alpha = pcmci_alpha, 
                    min_lag = min_lag, 
                    max_lag = max_lag, 
                    sel_method = TE(TEestimator.Gaussian), 
                    val_condtest = GPDC(significance = 'analytic', gp_params = None),
                    verbosity = CPLevel.DEBUG,
                    neglect_only_autodep = True)
    
    cm = fpcmci.run()
    fpcmci.dag(label_type = LabelType.Lag, node_layout = 'dot')
    fpcmci.timeseries_dag()
    
    # dynotears = DYNOTEARS(alpha = pcmci_alpha,
    #                       max_lag = max_lag)
    # cm = dynotears.run(df.d)
    # cm.dag()