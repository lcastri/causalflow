import copy
import datetime
from enum import Enum
import json
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

_TIME = 'time'
_PREC = 'precision'
_RECA = 'recall'
_F1SCORE = 'f1_score'
_SHD = "shd"
_FN = "fn"
_FP = "fp"
_TP = "tp"
_FPR = "fpr"
_PCMCI = 'pcmci'
_FPCMCI = 'fpcmci'
_doFPCMCI = 'dofpcmci'
_SCM = 'scm'
_N_ESPU = 'N_SpuriousLinks'
_N_GSPU = 'N_ExpectedSpuriousLinks'

dlabel = {_TIME : 'Time [s]',
          _PREC : 'Precision',
          _RECA : 'Recall',
          _F1SCORE : 'F1-score',
          _SHD : 'SHD',
          _PCMCI : 'PCMCI',
          _FPCMCI : 'FPCMCI',
          _doFPCMCI : 'doFPCMCI',
          _FPR: 'False Positive Rate',
          _N_ESPU : '# Sp. links estimated / # Sp. links generated'}


class plotType(Enum):
    Line = 0
    LinewErrorBar = 1
    LinewErrorBand = 2
    BoxPlot = 3
    
class ExtractDataMode(Enum):
    All = 0
    MeandStd = 1
    BootStrap = 2
    
def extract_data(file_path, data, mode = ExtractDataMode.MeandStd):
    since = datetime.datetime(1900, 1, 1, 0, 0, 0, 0)
    with open(file_path) as json_file:
        r = json.load(json_file)
        data_fpcmci = list()
        data_dofpcmci = list()
        for i in r.keys():
            if data == _TIME:
                time_fpcmci = datetime.datetime.strptime(r[i][_FPCMCI][data], '%H:%M:%S.%f')
                time_dofpcmci = datetime.datetime.strptime(r[i][_doFPCMCI][data], '%H:%M:%S.%f')
                data_fpcmci.append((time_fpcmci - since).total_seconds())
                data_dofpcmci.append((time_dofpcmci - since).total_seconds())
                
            elif data == _N_ESPU:
                data_fpcmci.append((r[i][_FPCMCI][data])/r[i][_N_GSPU])
                data_dofpcmci.append((r[i][_doFPCMCI][data])/r[i][_N_GSPU])
                
            else:
                data_fpcmci.append(r[i][_FPCMCI][data])
                data_dofpcmci.append(r[i][_doFPCMCI][data])
                
    if mode == ExtractDataMode.MeandStd:
        return np.mean(data_fpcmci), np.std(data_fpcmci), np.mean(data_dofpcmci), np.std(data_dofpcmci)
    
    elif mode == ExtractDataMode.BootStrap: 
        lower_bound, upper_bound = confidence_interval(data_fpcmci)
        fpcmci_conf = (upper_bound - lower_bound) / 2
        
        lower_bound, upper_bound = confidence_interval(data_dofpcmci)
        dofpcmci_conf = (upper_bound - lower_bound) / 2
        return np.mean(data_fpcmci), fpcmci_conf, np.mean(data_dofpcmci), dofpcmci_conf
        
    elif mode == ExtractDataMode.All:
        return data_fpcmci, data_dofpcmci
        
          
def plot_boxplot(resfolder, data, nvars, show = False):
    fpcmci_values = list()
    dofpcmci_values = list()
    
    for n in range(nvars[0],nvars[1]+1):
        res_path = os.getcwd() + "/results/" + resfolder + "/" + str(n) + ".json"
        fpcmci_data, dofpcmci_data = extract_data(res_path, data, returnMeanStd = False)
        fpcmci_values += [fpcmci_data]
        dofpcmci_values += [dofpcmci_data]
        
        
    fig, ax1 = plt.subplots(figsize=(9,6))
    box_param = dict(whis=(5, 95), widths=0.2, patch_artist=True, medianprops=dict(color='black'))
    space = 0.15

    dofpcmci_bp = ax1.boxplot(dofpcmci_values, positions=np.arange(nvars[0], nvars[1] + 1) - space,
                              boxprops=dict(facecolor='tab:blue', edgecolor='tab:blue', linewidth=1),
                              flierprops=dict(marker='.', markeredgecolor='tab:blue', fillstyle=None), **box_param)

    fpcmci_bp = ax1.boxplot(fpcmci_values, positions=np.arange(nvars[0], nvars[1] + 1) + space,
                              boxprops=dict(facecolor='tab:red', edgecolor='tab:red', linewidth=1),
                              flierprops=dict(marker='.', markeredgecolor='tab:red', fillstyle=None), **box_param)
    
    # if data in [_F1SCORE, _PREC, _RECA]: plt.ylim(0, 1.1)
        
    ax1.set_xticks(np.arange(nvars[0],nvars[1]+1))
    ax1.set_xticklabels(np.arange(nvars[0],nvars[1]+1))

    ax1.set_xlabel("# vars")
    ax1.set_ylabel(dlabel[data])

    ax1.grid()
    ax1.legend([dofpcmci_bp["boxes"][0], fpcmci_bp["boxes"][0]], [dlabel[_doFPCMCI], dlabel[_FPCMCI]], loc='best')

    plt.title(data + ' comparison')
    
    if show:
        plt.show()
    else:
        plt.savefig(os.getcwd() + "/results/" + resfolder + "/" + data + '.pdf')
        plt.savefig(os.getcwd() + "/results/" + resfolder + "/" + data + '.png')
    
    
def compare(resfolder, data, nvars, plot_type = plotType.LinewErrorBar, bootStrap = False, show = False):
    if plot_type == plotType.BoxPlot:
       plot_boxplot(resfolder, data, nvars)
       return
   
    fpcmci_means = list()
    fpcmci_stds = list()
    dofpcmci_means = list()
    dofpcmci_stds = list()
    
    for n in range(nvars[0],nvars[1]+1):
        res_path = os.getcwd() + "/results/" + resfolder + "/" + str(n) + ".json"
        
        fpcmci_mean, fpcmci_std, dofpcmci_mean, dofpcmci_std = extract_data(res_path, data, mode = ExtractDataMode.MeandStd if not bootStrap else ExtractDataMode.BootStrap)
        fpcmci_means.append(fpcmci_mean)
        fpcmci_stds.append(fpcmci_std)
        dofpcmci_means.append(dofpcmci_mean)
        dofpcmci_stds.append(dofpcmci_std)
            
    fig, ax = plt.subplots(figsize=(6,4))
    
    if plot_type == plotType.Line:
        plt.plot(range(nvars[0], nvars[1]+1), dofpcmci_means)
        plt.plot(range(nvars[0], nvars[1]+1), fpcmci_means)
        
    elif plot_type == plotType.LinewErrorBar:
        plt.errorbar(range(nvars[0], nvars[1]+1), dofpcmci_means, dofpcmci_stds, marker='o', capsize = 5, color = 'b')
        plt.errorbar(range(nvars[0], nvars[1]+1), fpcmci_means, fpcmci_stds, marker='^', capsize = 5, color = 'r', linestyle = '--')
        
    elif plot_type == plotType.LinewErrorBand:
        plt.plot(range(nvars[0], nvars[1]+1), dofpcmci_means, marker='o', color = 'b')
        plt.plot(range(nvars[0], nvars[1]+1), fpcmci_means, marker='^', color = 'r', linestyle = '--')
        plt.fill_between(range(nvars[0], nvars[1]+1), np.array(dofpcmci_means) - np.array(dofpcmci_stds), np.array(dofpcmci_means) + np.array(dofpcmci_stds), alpha=0.3, color = 'b')
        plt.fill_between(range(nvars[0], nvars[1]+1), np.array(fpcmci_means) - np.array(fpcmci_stds), np.array(fpcmci_means) + np.array(fpcmci_stds), alpha=0.3, color = 'r')
    
    plt.xticks(range(nvars[0], nvars[1]+1))
    # if data in [_F1SCORE, _PREC, _RECA]: plt.ylim(0, 1.1)
    plt.xlabel("# vars")
    plt.ylabel(dlabel[data])
    plt.legend([dlabel[_doFPCMCI], dlabel[_FPCMCI]])
    plt.grid()
    plt.title(data + ' comparison')
    
    if show:
        plt.show()
    else:
        plt.savefig(os.getcwd() + "/results/" + resfolder + "/" + data + '.pdf')
        plt.savefig(os.getcwd() + "/results/" + resfolder + "/" + data + '.png')
    
    
def plot_distribution(resfolder, data, nvars):
    values = list()
    for n in range(nvars[0], nvars[1]+1):
        res_path = os.getcwd() + "/results/" + resfolder + "/" + str(n) + ".json"
        since = datetime.datetime(1900, 1, 1, 0, 0, 0, 0)
        with open(res_path) as json_file:
            r = json.load(json_file)
            for i in r.keys():
                if data == _TIME:
                    time_tepcmci = datetime.datetime.strptime(r[i][_FPCMCI][data], '%H:%M:%S.%f')
                    time_pcmci = datetime.datetime.strptime(r[i][_doFPCMCI][data], '%H:%M:%S.%f')
                    values.append((time_tepcmci - since).total_seconds())
                    values.append((time_pcmci - since).total_seconds())
                else:
                    values.append(r[i][_FPCMCI][data])
                    values.append(r[i][_doFPCMCI][data])
                    
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Create a histogram
    ax.hist(values, bins=20)

    # Set labels and title
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram')

    # Show the plot
    plt.show()


def confidence_interval(data, confidence_level=0.95, n_resamples = 1000):
    """
    Calculate confidence intervals for a given dataset.
    
    Parameters:
        data (array-like): Array of data for which confidence intervals are calculated.
        confidence_level (float, optional): Confidence level for the intervals (default: 0.95).
    
    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """
    # Number of data points
    n = len(data)
    
    # Initialize array to store bootstrap resampled statistics
    resample_stats = np.zeros(n_resamples)
    
    # Perform bootstrap resampling
    for i in range(n_resamples):
        resample = np.random.choice(data, size=n, replace=True)
        resample_stats[i] = np.mean(resample)  # Calculate the statistic of interest
    
    # Calculate confidence interval
    lower = np.percentile(resample_stats, (1 - confidence_level) / 2 * 100)
    upper = np.percentile(resample_stats, (1 + confidence_level) / 2 * 100)
    
    return lower, upper    

    
if __name__ == '__main__':   

    resfolder = ['good/nvariable_1hconf_nonlin_1000_1000_0_0.5.5']
    vars = [7, 14]
    bootstrap = True
    for r in resfolder:
        for metric in [_TIME,_F1SCORE, _PREC, _RECA, _SHD, _FPR, _N_ESPU]:
            compare(r, metric, vars, plotType.LinewErrorBar, bootStrap = bootstrap, x_label = '# vars')
        # compare(r, _F1SCORE, vars, plotType.LinewErrorBar, bootStrap = bootstrap)
        # compare(r, _PREC, vars, plotType.LinewErrorBar, bootStrap = bootstrap)
        # compare(r, _RECA, vars, plotType.LinewErrorBar, bootStrap = bootstrap)
        # compare(r, _SHD, vars, plotType.LinewErrorBar, bootStrap=bootstrap)
        # compare(r, _FPR, vars, plotType.LinewErrorBar, bootStrap=bootstrap)
        # compare(r, "N_SpuriousLinks", vars, plotType.LinewErrorBar, bootStrap=bootstrap)