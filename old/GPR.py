from copy import deepcopy
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
import matplotlib.pyplot as plt
import numpy as np
from fpcmci.FPCMCI import FPCMCI
from fpcmci.CPrinter import CPLevel
from fpcmci.basics.constants import LabelType
from fpcmci.preprocessing.data import Data
from fpcmci.selection_methods.TE import TE, TEestimator
from tigramite.independence_tests.gpdc import GPDC
import dcor


def measure_dist(x, y):
    observed_dcor = dcor.distance_correlation(x, y)

    # Number of permutations for the test
    num_permutations = 1000

    # Perform permutation test
    permuted_dcors = []
    for _ in range(num_permutations):
        # Permute one of the time series
        permuted_y = np.random.permutation(y)
        # Calculate distance correlation with permuted series
        permuted_dcor = dcor.distance_correlation(x, permuted_y)
        permuted_dcors.append(permuted_dcor)

    # Calculate p-value as the proportion of permuted distances greater than or equal to the observed distance
    p_value = (np.sum(permuted_dcors >= observed_dcor) + 1) / (num_permutations + 1)
    
    return observed_dcor, p_value


if __name__ == '__main__':   

    resdir = "FPCMCI_vs_doFPCMCI_simple"
    f_alpha = 0.1
    pcmci_alpha = 0.05
    min_lag = 1
    max_lag = 2

    np.random.seed(1)
    nsample = 1500
    Tint = nsample
    nfeature = 5
    d = np.random.random(size = (nsample, nfeature))
    for t in range(max_lag, nsample):
        d[t, 0] += 0.15 * d[t-1, 0]
        d[t, 1] += 0.5 * d[t-1, 0]**2
        d[t, 2] += 0.75 * d[t-1, 1] + 0.9 * d[t-2, 4] 
        d[t, 3] += 0.33 * d[t-1, 4] * d[t-1, 1] 
       
    df_obs = Data(d, vars = ['A', 'B', 'C', 'D', 'E'])
    
    df_obs_hidden = deepcopy(df_obs)
    df_obs_hidden.shrink(['A', 'B', 'C', 'D'])
    
    
    int_data = dict()
    # D
    d_intD = np.random.random(size = (Tint, nfeature))
    d_intD[:, 3] = 15 * np.ones(shape = (Tint,)) 
    for t in range(max_lag, Tint):
        d_intD[t, 0] += 0.15 * d_intD[t-1, 0]
        d_intD[t, 1] += 0.5 * d_intD[t-1, 0]**2
        d_intD[t, 2] += 0.75 * d_intD[t-1, 1] + 0.9 * d_intD[t-2, 4] + 1.2 * d_intD[t-1, 3] 
        
    df_int = Data(d_intD, vars = ['A', 'B', 'C', 'D', 'E'])
    df_int.shrink(['A', 'B', 'C', 'D'])
    int_data['D'] =  df_int
    
    
    # FPCMCI
    fpcmci = FPCMCI(deepcopy(df_obs_hidden),
                    f_alpha = f_alpha, 
                    pcmci_alpha = pcmci_alpha, 
                    min_lag = min_lag, 
                    max_lag = max_lag, 
                    sel_method = TE(TEestimator.Gaussian), 
                    val_condtest = GPDC(significance = 'analytic', gp_params = None),
                    verbosity = CPLevel.INFO,
                    neglect_only_autodep = False,
                    resfolder = resdir + "/fpcmci_hidden")

    startFPCMCI = datetime.now()
    features, cm = fpcmci.run()
    stopFPCMCI = datetime.now()  
    fpcmci.dag(node_layout='circular', label_type = LabelType.Lag)
    fpcmci.timeseries_dag()

    # GPR ###################################################################################################
    for int_var in int_data:
        for t in cm.g:
            for s in cm.g[t].sources:
                if int_var != s[0]: 
                    continue
                else:
                    print("Testing (" + str(s[0]) + " - " + str(s[1]) + ") --> (" + str(t) + ")")
                    
                    # Create kernel and define GPR
                    kernel = RBF() + WhiteKernel()
                    gpr = GaussianProcessRegressor(kernel = kernel)

                    # Prepare observational data
                    tmp_obs = deepcopy(df_obs_hidden)
                    Y_obs = tmp_obs.d[t].values
                    tmp_obs.shrink(cm.g[t].sourcelist)
                    X_obs = tmp_obs.d.values
                    
                    # Fit GPR model with observational data
                    gpr.fit(X_obs, Y_obs)

                    # Prepare interventional data
                    tmp_int = deepcopy(int_data[int_var])
                    Y_int = tmp_int.d[t].values
                    tmp_int.shrink(cm.g[t].sourcelist)
                    X_int = tmp_int.d.values
                    
                    # Predict mean
                    y_hat = gpr.predict(X_int)

                    # Calculate distance correlation and p-value
                    dcor_value, p_value = measure_dist(Y_obs, y_hat)
                    print("Distance correlation:", dcor_value)
                    print("p-value:", p_value)
                    if p_value < pcmci_alpha:
                        print("ok")
                    else:
                        print("link to remove")
                         
                    # Initialize plot
                    f, ax = plt.subplots(1, 1, figsize=(4, 3))

                    Y_gt = np.concatenate((Y_obs, Y_int), axis = 0)
                    Y_hat = np.concatenate((Y_obs, y_hat), axis = 0)

                    # Plot ground truth and predictive means
                    ax.plot(range(len(Y_gt)), Y_hat, 'r', label = "predicted")
                    ax.plot(range(len(Y_gt)), Y_gt, 'b', label = "observed")
                    ax.axvline(x = len(Y_obs), color = 'gray', ls = '--', lw = 3)
                    
                    # Shade between the lower and upper confidence bounds
                    plt.show()