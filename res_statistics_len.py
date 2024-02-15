import datetime
from enum import Enum
import json
import os
from matplotlib import pyplot as plt
import numpy as np

class jWord(Enum):
    SCM = 'scm'
    N_GSPU = 'N_ExpectedSpuriousLinks'
    GT = "GT"
    Confounders = "Confounders"
    HiddenConfounders = "HiddenConfounders"
    InterventionVariables = "InterventionVariables"
    ExpectedSpuriousLinks = "ExpectedSpuriousLinks"
    SpuriousLinks = "SpuriousLinks"
    
class Algo(Enum):
    CAnDOIT = 'candoit'
    CAnDOITLagged = 'candoit_lagged'
    CAnDOITCont = 'candoit_cont'
    DYNOTEARS = 'dynotears'
    FPCMCI = 'fpcmci'
    PCMCI = 'pcmci'
    TCDF = 'tcdf'
    tsFCI = 'tsfci'
    VarLiNGAM = 'varlingam'
    
class Metric(Enum):
    FN = "fn"
    FP = "fp"
    TP = "tp"
    TIME = 'time'
    F1SCORE = 'f1_score'
    PREC = 'precision'
    RECA = 'recall'
    SHD = "shd"
    FPR = "fpr"
    N_ESPU = 'N_SpuriousLinks'
    N_EqDAG = 'N_EquiDAG'

plotLabel = {Metric.TIME : 'Time [s]',
             Metric.PREC : 'Precision',
             Metric.RECA : 'Recall',
             Metric.F1SCORE : '$F_1$ Score',
             Metric.SHD : 'SHD',
             Algo.CAnDOIT : 'CAnDOIT',
             Algo.CAnDOITCont : 'CAnDOITCont',
             Algo.CAnDOITLagged : 'CAnDOITLagged',
             Algo.DYNOTEARS : 'DYNOTEARS',
             Algo.FPCMCI : 'F-PCMCI',
             Algo.PCMCI : 'PCMCI',
             Algo.TCDF : 'TCDF',
             Algo.tsFCI : 'tsFCI',
             Algo.VarLiNGAM : 'VarLiNGAM',
             Metric.FPR: 'False Positive Rate',
             Metric.N_ESPU : '# Sp. links estimated / # Sp. links',
             Metric.N_EqDAG : "# Equ. DAGs"}

titleLabel = {Metric.TIME : 'Time',
              Metric.PREC : 'Precision',
              Metric.RECA : 'Recall',
              Metric.F1SCORE : '$F_1$ Score',
              Metric.SHD : 'SHD',
              Metric.FPR: 'FPR',
              Metric.N_ESPU : 'Spurious Links',
              Metric.N_EqDAG : '# Equivalent DAGs',}

class plotType(Enum):
    Line = 0
    LinewErrorBar = 1
    LinewErrorBand = 2
    BoxPlot = 3    
    
class ExtractDataMode(Enum):
    MeandStd = 0
    BootStrap = 1
    
    
def extract_data(metric, mode = ExtractDataMode.MeandStd):
    ext_data = {str(round(perc/100,2)) + "_" + str(round(1 - perc/100, 2)): {"samples" : list(), "mean" : float, "confidence" : float} for perc in range(5, 100, 5)}
    since = datetime.datetime(1900, 1, 1, 0, 0, 0, 0)
    
    for n in range(25):
        file_path = os.getcwd() + "/results/CAnDOIT_bestlength/" + str(n) + ".json"
        with open(file_path) as json_file:
            r = json.load(json_file)
            
            for i in r.keys():
                if metric == Metric.TIME:
                    t = datetime.datetime.strptime(r[i][Algo.CAnDOITCont.value][metric.value], '%H:%M:%S.%f')
                    ext_data[i]["samples"].append((t - since).total_seconds())
                    
                elif metric == Metric.N_ESPU:
                    if r[i][jWord.N_GSPU.value] != 0:
                        ext_data[i]["samples"].append((r[i][Algo.CAnDOITCont.value][metric.value])/r[i][jWord.N_GSPU.value])
                    else:
                        ext_data[i]["samples"].append(0)
                else:
                    ext_data[i]["samples"].append((r[i][Algo.CAnDOITCont.value][metric.value]))
    for i in r.keys():
        if mode == ExtractDataMode.MeandStd:
            ext_data[i]["mean"] = np.mean(ext_data[i]["samples"])
            ext_data[i]["confidence"] = np.std(ext_data[i]["samples"])
        
        elif mode == ExtractDataMode.BootStrap: 
            ext_data[i]["mean"] = float(np.mean(ext_data[i]["samples"]))
            lower_bound, upper_bound = confidence_interval(ext_data[i]["samples"])
            ext_data[i]["confidence"] = float((upper_bound - lower_bound) / 2)
        
    return ext_data
           
    
def compare(resfolder, metric, plotStyle, bootStrap = False, show = False, xLabel = 'obs/int perc'):
   
    toPlot = {"xLabel" : list(), "means" : list(), "confidences" : list()}

    ext_data = extract_data(metric, mode = ExtractDataMode.MeandStd if not bootStrap else ExtractDataMode.BootStrap)
    for i in ext_data.keys():
        toPlot["xLabel"].append(i.split("_")[0])
        toPlot["means"].append(ext_data[i]["mean"])
        toPlot["confidences"].append(ext_data[i]["confidence"])
    
    toPlot["xLabel"].reverse()
    toPlot["means"].reverse()
    toPlot["confidences"].reverse()
    
    fig, ax1 = plt.subplots(figsize=(9,6))
    plt.errorbar(range(len(toPlot["means"])), toPlot["means"], toPlot["confidences"], 
                 marker=plotStyle[Algo.CAnDOITCont]['marker'], 
                 capsize = 5, 
                 color = plotStyle[Algo.CAnDOITCont]['color'], 
                 linestyle = plotStyle[Algo.CAnDOITCont]['linestyle'])
    plt.xticks(ticks=range(len(toPlot["means"])), labels=toPlot["xLabel"])
    plt.xlabel(xLabel)
    plt.ylabel(plotLabel[metric])
    plt.grid()
             
    if show:
        plt.show()
    else:
        plt.savefig(os.getcwd() + "/results/" + resfolder + "/" + metric.value + '.pdf')
        plt.savefig(os.getcwd() + "/results/" + resfolder + "/" + metric.value + '.png')
        
        
def complete(resfolder, metrics, bootStrap = False, show = False, xLabel = 'obs/int perc'):
   
    toPlot = {m: {"xLabel" : list(), "means" : list(), "confidences" : list()} for m in metrics}

    for m in metrics:
        ext_data = extract_data(m, mode = ExtractDataMode.MeandStd if not bootStrap else ExtractDataMode.BootStrap)
        for i in ext_data.keys():
            toPlot[m]["xLabel"].append(i.split("_")[0])
            toPlot[m]["means"].append(ext_data[i]["mean"])
            toPlot[m]["confidences"].append(ext_data[i]["confidence"])
        
        toPlot[m]["xLabel"].reverse()
        toPlot[m]["means"].reverse()
        toPlot[m]["confidences"].reverse()

        toPlot[m]["means"] = [c/max(toPlot[m]["means"]) for c in toPlot[m]["means"]]
        toPlot[m]["confidences"] = [c/max(toPlot[m]["confidences"]) for c in toPlot[m]["confidences"]]
            
    
    fig, ax1 = plt.subplots(figsize=(9,6))
    for m in metrics:
        plt.plot(range(len(toPlot[m]["means"])), toPlot[m]["means"], label = m.value)
    plt.xticks(ticks=range(len(toPlot[m]["means"])), labels=toPlot[m]["xLabel"])
    plt.legend()
    plt.grid()
    plt.xlabel(xLabel)
             
    if show:
        plt.show()
    else:
        plt.savefig(os.getcwd() + "/results/" + resfolder + "/overall.pdf")
        plt.savefig(os.getcwd() + "/results/" + resfolder + "/overall.png")


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

    resfolder = 'CAnDOIT_bestlength'
    
    bootstrap = True
    algorithms = [Algo.CAnDOITCont]
    plot_style = {Algo.CAnDOITCont: {"marker" : 'o', "color" : 'b', "linestyle" : '-'},}
    # for metric in [Metric.TIME, Metric.F1SCORE, Metric.PREC, Metric.RECA, Metric.SHD, Metric.FPR, Metric.N_ESPU, Metric.N_EqDAG]:
    #     compare(resfolder, metric, plot_style, bootStrap = bootstrap)
    complete(resfolder, [Metric.TIME, Metric.F1SCORE, Metric.PREC, Metric.RECA, Metric.SHD, Metric.FPR, Metric.N_ESPU, Metric.N_EqDAG], bootStrap = bootstrap)