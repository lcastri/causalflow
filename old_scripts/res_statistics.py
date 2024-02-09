import datetime
from enum import Enum
import json
import os
from matplotlib import pyplot as plt
import numpy as np

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
_CAnDOIT = 'candoit'
_SCM = 'scm'
_N_ESPU = 'N_SpuriousLinks'
_N_EqDAG= 'N_EquiDAG_2exp'
_N_GSPU = 'N_ExpectedSpuriousLinks'


yLabel = {_TIME : 'Time [s]',
          _PREC : 'Precision',
          _RECA : 'Recall',
          _F1SCORE : 'F1-score',
          _SHD : 'SHD',
          _PCMCI : 'PCMCI',
          _FPCMCI : 'FPCMCI',
          _CAnDOIT : 'CAnDOIT',
          _FPR: 'False Positive Rate',
          _N_ESPU : '# Sp. links estimated / # Sp. links',
          _N_EqDAG : "# Equ. DAGs"}


titleLabel = {_TIME : 'Time',
              _PREC : 'Precision',
              _RECA : 'Recall',
              _F1SCORE : 'F1-score',
              _SHD : 'SHD',
              _FPR: 'FPR',
              _N_ESPU : 'Spurious Links',
              _N_EqDAG : '# Equivalent DAGs',}


class plotType(Enum):
    Line = 0
    LinewErrorBar = 1
    LinewErrorBand = 2
    BoxPlot = 3
    
    
class ExtractDataMode(Enum):
    MeandStd = 0
    BootStrap = 1
    
    
def extract_data(file_path, algorithms, data, mode = ExtractDataMode.MeandStd):
    ext_data = {algo: {"samples" : list(), "mean" : float, "confidence" : float} for algo in algorithms}

    since = datetime.datetime(1900, 1, 1, 0, 0, 0, 0)
    with open(file_path) as json_file:
        r = json.load(json_file)
        
        for i in r.keys():
            if data == _TIME:
                for algo in algorithms:
                    t = datetime.datetime.strptime(r[i][algo][data], '%H:%M:%S.%f')
                    ext_data[algo]["samples"].append((t - since).total_seconds())
                
            elif data == _N_ESPU:
                for algo in algorithms:
                    if r[i][_N_GSPU] != 0:
                        ext_data[algo]["samples"].append((r[i][algo][data])/r[i][_N_GSPU])
                    else:
                        ext_data[algo]["samples"].append(0)
                
            else:
                for algo in algorithms:
                    ext_data[algo]["samples"].append((r[i][algo][data]))
                
    if mode == ExtractDataMode.MeandStd:
        for algo in algorithms:
            ext_data[algo]["mean"] = np.mean(ext_data[algo]["samples"])
            ext_data[algo]["confidence"] = np.std(ext_data[algo]["samples"])
    
    elif mode == ExtractDataMode.BootStrap: 
        for algo in algorithms:
            ext_data[algo]["mean"] = np.mean(ext_data[algo]["samples"])
            lower_bound, upper_bound = confidence_interval(ext_data[algo]["samples"])
            ext_data[algo]["confidence"] = (upper_bound - lower_bound) / 2
        
    return ext_data
           
    
def compare(resfolder, algorithms, data, nvars, plotStyle, plot_type = plotType.LinewErrorBar, bootStrap = False, show = False, xLabel = '# vars'):
   
    toPlot = {algo: {"samples" : list(), "means" : list(), "confidences" : list()} for algo in algorithms}
    
    for n in range(nvars[0],nvars[1]+1):
        res_path = os.getcwd() + "/results/" + resfolder + "/" + str(n) + ".json"
        
        ext_data = extract_data(res_path, algorithms, data, mode = ExtractDataMode.MeandStd if not bootStrap else ExtractDataMode.BootStrap)
        for algo in algorithms:
            toPlot[algo]["samples"].append(ext_data[algo]["samples"])
            toPlot[algo]["means"].append(ext_data[algo]["mean"])
            toPlot[algo]["confidences"].append(ext_data[algo]["confidence"])

    # print("Score: " + str(data))
    # for algo in algorithms:
    #     stri = list()
    #     print("Algorithm: " + str(algo))
    #     for m, c in zip(toPlot[algo]["means"], toPlot[algo]["confidences"]):
    #         stri.append(str(round(m,3)) + "\u00B1" + str(round(c,3)))
    #         # print("| " + str(round(m,3)) + "\u00B1" + str(round(c,3)) + " |")
    #     print(" | ".join(stri))
            
    if plot_type != plotType.BoxPlot:
        fig, ax = plt.subplots(figsize=(6,4))

        if plot_type == plotType.Line:
            for algo in algorithms:
                plt.plot(range(nvars[0], nvars[1]+1), toPlot[algo]["means"], 
                        marker=plotStyle[algo]['marker'], color = plotStyle[algo]['color'], linestyle = plotStyle[algo]['linestyle'])
            
        elif plot_type == plotType.LinewErrorBar:
            for algo in algorithms:
                plt.errorbar(range(nvars[0], nvars[1]+1), toPlot[algo]["means"], toPlot[algo]["confidences"], 
                            marker=plotStyle[algo]['marker'], capsize = 5, color = plotStyle[algo]['color'], linestyle = plotStyle[algo]['linestyle'])
            
        elif plot_type == plotType.LinewErrorBand:
            plt.plot(range(nvars[0], nvars[1]+1), toPlot[algo]["means"],
                    marker=plotStyle[algo]['marker'], color = plotStyle[algo]['color'], linestyle = plotStyle[algo]['linestyle'])
            plt.fill_between(range(nvars[0], nvars[1]+1), 
                            np.array(toPlot[algo]["means"]) - np.array(toPlot[algo]["confidences"]), 
                            np.array(toPlot[algo]["means"]) + np.array(toPlot[algo]["confidences"]), 
                            alpha=0.3, color = plotStyle[algo]['color'])
        
        plt.xticks(range(nvars[0], nvars[1]+1))
        plt.xlabel(xLabel)
        plt.ylabel(yLabel[data])
        plt.legend([yLabel[algo] for algo in algorithms], loc='best')
        plt.grid()
        plt.title(titleLabel[data] + ' comparison')
        
    else:
        fig, ax1 = plt.subplots(figsize=(9,6))
        box_width = 0.2
        space = 0.15
        N = len(algorithms)
        total_width = N * (box_width + space)
        positions = np.arange(nvars[1] - nvars[0] + 1) * total_width

        # positions = np.arange(len(np.arange(nvars[0], nvars[1]+1))) * (N * (box_width + space)) + (N - 1) * (box_width + space) / 2

        boxplots = list()
        box_param = dict(whis=(5, 95), widths=box_width, patch_artist=True, medianprops=dict(color='black'))
        
        
        for i in range(nvars[1] - nvars[0] + 1):
            for j, algo in enumerate(algorithms):
                position = positions[i] + j * (box_width + space) + box_width/2
                boxplots.append(ax1.boxplot(toPlot[algo]["samples"][i], positions=[position],
                                            boxprops=dict(facecolor=plotStyle[algo]['color'], edgecolor=plotStyle[algo]['color'], linewidth=1),
                                            flierprops=dict(marker='.', markeredgecolor=plotStyle[algo]['color'], fillstyle=None), **box_param))
            
        # ax1.set_xticks(np.arange(nvars[0],nvars[1]+1))
        ax1.set_xticks(positions + (total_width - space) / 2)
        ax1.set_xticklabels(np.arange(nvars[0],nvars[1]+1))

        ax1.set_xlabel(xLabel)
        ax1.set_ylabel(yLabel[data])

        ax1.grid()
        ax1.legend([bp['boxes'][0] for bp in boxplots], [yLabel[algo] for algo in algorithms], loc='best')
         
    # if show:
    #     plt.show()
    # else:
    #     plt.savefig(os.getcwd() + "/results/" + resfolder + "/" + data + '.pdf')
    #     plt.savefig(os.getcwd() + "/results/" + resfolder + "/" + data + '.png')


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

    # To use to plot RS_comparison_variables
    # resfolder = ['nvariable_1hconf_nonlin_1000_1000_0_0.5']
    # vars = [7, 14]
    
    
    # To use to plot RS_comparison_nconfounded
    resfolder = ['rebuttal/nconfounded_nonlin_1250_250']
    vars = [0, 7]
    # resfolder = ['rebuttal/nvariable_1hconf_nonlin_1250_250']
    # vars = [7, 14]
    
    
    bootstrap = True
    algorithms = [_PCMCI, _FPCMCI, _CAnDOIT]
    plot_style = {_PCMCI: {"marker" : 'x', "color" : 'g', "linestyle" : ':'},
                  _FPCMCI: {"marker" : '^', "color" : 'r', "linestyle" : '--'},
                  _CAnDOIT: {"marker" : 'o', "color" : 'b', "linestyle" : '-'}}
    for r in resfolder:
        for metric in [_TIME,_F1SCORE, _PREC, _RECA, _SHD, _FPR, _N_ESPU, _N_EqDAG]:
            compare(r, algorithms, metric, vars, plot_style, plotType.LinewErrorBar, bootStrap = bootstrap, xLabel = '# confounded vars')
            # compare(r, algorithms, metric, vars, plot_style, plotType.LinewErrorBar, bootStrap = bootstrap)