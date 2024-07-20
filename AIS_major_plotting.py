import datetime
from enum import Enum
import json
import os
from matplotlib import pyplot as plt
import numpy as np

    
class Algo(Enum):
    CAnDOIT = 'candoit'
    DYNOTEARS = 'dynotears'
    FPCMCI = 'fpcmci'
    PCMCI = 'pcmci'
    PCMCIplus = 'pcmciplus'
    LPCMCI = 'lpcmci'
    TCDF = 'tcdf'
    tsFCI = 'tsfci'
    VarLiNGAM = 'varlingam'
    
class Metric(Enum):
    ADJ_FN = "adj_fn"
    ADJ_FP = "adj_fp"
    ADJ_TP = "adj_tp"
    ADJ_TN = "adj_tn"
    ADJ_F1SCORE = 'adj_f1_score'
    ADJ_PREC = 'adj_precision'
    ADJ_RECA = 'adj_recall'
    ADJ_SHD = "adj_shd"
    ADJ_FPR = "adj_fpr"
    ADJ_TPR = "adj_tpr"
    ADJ_FNR = "adj_fnr"
    ADJ_TNR = "adj_tnr"
    GRAPH_FN = "graph_fn"
    GRAPH_FP = "graph_fp"
    GRAPH_TP = "graph_tp"
    GRAPH_TN = "graph_tn"
    GRAPH_F1SCORE = 'graph_f1_score'
    GRAPH_PREC = 'graph_precision'
    GRAPH_RECA = 'graph_recall'
    GRAPH_SHD = "graph_shd"
    GRAPH_FPR = "graph_fpr"
    GRAPH_TPR = "graph_tpr"
    GRAPH_FNR = "graph_fnr"
    GRAPH_TNR = "graph_tnr"
    TIME = 'time'
    UNCERTAINTY = 'uncertainty'
    PAGSIZE = 'pag_size'

plotLabel = {Metric.TIME : 'Time [s]',
             Metric.ADJ_PREC : 'Adjacency Precision',
             Metric.ADJ_RECA : 'Adjacency Recall',
             Metric.ADJ_F1SCORE : 'Adjacency $F_1$ Score',
             Metric.ADJ_SHD : 'Adjacency SHD',
             Metric.ADJ_FPR: 'Adjacency False Positive Rate',
             Metric.GRAPH_PREC : 'Adjacency and Orientation Precision',
             Metric.GRAPH_RECA : 'Adjacency and Orientation Recall',
             Metric.GRAPH_F1SCORE : 'Adjacency and Orientation $F_1$ Score',
             Metric.GRAPH_SHD : 'Adjacency and Orientation SHD',
             Metric.GRAPH_FPR: 'Adjacency and Orientation False Positive Rate',
             Metric.UNCERTAINTY : 'Uncertainty',
             Metric.PAGSIZE : "PAG Size",
             'candoit_best' : 'CAnDOIT_best',
             'candoit_mean' : 'CAnDOIT_mean',
             Algo.DYNOTEARS : 'DYNOTEARS',
             Algo.FPCMCI : 'F-PCMCI',
             Algo.PCMCI : 'PCMCI',
             Algo.LPCMCI : 'LPCMCI',
             'lpcmci' : 'LPCMCI',
             Algo.TCDF : 'TCDF',
             Algo.tsFCI : 'tsFCI',
             Algo.VarLiNGAM : 'VarLiNGAM'}

titleLabel = {Metric.TIME : 'Time',
              Metric.ADJ_PREC : 'Adjacency Precision',
              Metric.ADJ_RECA : 'Adjacency Recall',
              Metric.ADJ_F1SCORE : 'Adjacency $F_1$ Score',
              Metric.ADJ_SHD : 'Adjacency SHD',
              Metric.ADJ_FPR: 'Adjacency FPR',
              Metric.GRAPH_PREC : 'Adjacency and Orientation Precision',
              Metric.GRAPH_RECA : 'Adjacency and Orientation Recall',
              Metric.GRAPH_F1SCORE : 'Adjacency and Orientation $F_1$ Score',
              Metric.GRAPH_SHD : 'Adjacency and Orientation SHD',
              Metric.GRAPH_FPR: 'Adjacency and Orientation FPR',
              Metric.UNCERTAINTY : 'Uncertainty',
              Metric.PAGSIZE : '# Equivalent DAGs',}

class plotType(Enum):
    Line = 0
    LinewErrorBar = 1
    LinewErrorBand = 2
    BoxPlot = 3    
    
class ExtractDataMode(Enum):
    MeandStd = 0
    BootStrap = 1
    
    
def extract_data(file_path, algorithm, metric, mode = ExtractDataMode.MeandStd, candoit_mode = 'best'):
    # ext_data = {algo.value: {"samples" : list(), "mean" : float, "confidence" : float} for algo in algorithm}
    ext_data = {algorithm: {"samples" : list(), "mean" : float, "confidence" : float}}

    since = datetime.datetime(1900, 1, 1, 0, 0, 0, 0)
    with open(file_path) as json_file:
        r = json.load(json_file)
                
        for i in r.keys():
            candoit_keys = [key for key in r[i] if key.startswith('candoit__')]
            if candoit_mode == 'best': selected_candoit = candoit_keys[np.argmax([r[i][k]['graph_f1_score'] for k in candoit_keys])]
            if metric == Metric.TIME:
                # for algo in algorithm:
                # if algo == Algo.CAnDOIT:
                if algorithm == Algo.CAnDOIT.value:
                    if candoit_mode == 'best': 
                        t = datetime.datetime.strptime(r[i][selected_candoit][metric.value], '%H:%M:%S.%f')
                        # ext_data[algo.value]["samples"].append((t - since).total_seconds())
                        ext_data[algorithm]["samples"].append((t - since).total_seconds())
                    if candoit_mode == 'mean': 
                        ts = [datetime.datetime.strptime(r[i][candoit_run][metric.value], '%H:%M:%S.%f') for candoit_run in candoit_keys]
                        tmp = np.mean([(t - since).total_seconds() for t in ts])
                        # ext_data[algo.value]["samples"].append(tmp)
                        ext_data[algorithm]["samples"].append(tmp)
                else:
                    # t = datetime.datetime.strptime(r[i][algo.value][metric.value], '%H:%M:%S.%f')
                    # ext_data[algo.value]["samples"].append((t - since).total_seconds())
                    t = datetime.datetime.strptime(r[i][algorithm][metric.value], '%H:%M:%S.%f')
                    ext_data[algorithm]["samples"].append((t - since).total_seconds())
            else:
                # for algo in algorithm:
                # if algo == Algo.CAnDOIT:
                if algorithm == Algo.CAnDOIT.value:
                    if candoit_mode == 'best': 
                        # ext_data[algo.value]["samples"].append((r[i][selected_candoit][metric.value]))
                        ext_data[algorithm]["samples"].append((r[i][selected_candoit][metric.value]))
                    if candoit_mode == 'mean':
                        tmp = np.mean([r[i][candoit_run][metric.value] for candoit_run in candoit_keys])
                        # ext_data[algo.value]["samples"].append((tmp))
                        ext_data[algorithm]["samples"].append((tmp))
                else:
                    # ext_data[algo.value]["samples"].append((r[i][algo.value][metric.value]))
                    ext_data[algorithm]["samples"].append((r[i][algorithm][metric.value]))
                
    if mode == ExtractDataMode.MeandStd:
        # for algo in algorithm:
            # ext_data[algo.value]["mean"] = np.mean(ext_data[algo.value]["samples"])
            # ext_data[algo.value]["confidence"] = np.std(ext_data[algo.value]["samples"])
        ext_data[algorithm]["mean"] = np.mean(ext_data[algorithm]["samples"])
        ext_data[algorithm]["confidence"] = np.std(ext_data[algorithm]["samples"])
    
    elif mode == ExtractDataMode.BootStrap: 
        # for algo in algorithm:
            # ext_data[algo.value]["mean"] = np.mean(ext_data[algo.value]["samples"])
            # lower_bound, upper_bound = confidence_interval(ext_data[algo.value]["samples"])
            # ext_data[algo.value]["confidence"] = (upper_bound - lower_bound) / 2
        ext_data[algorithm]["mean"] = np.mean(ext_data[algorithm]["samples"])
        lower_bound, upper_bound = confidence_interval(ext_data[algorithm]["samples"])
        ext_data[algorithm]["confidence"] = (upper_bound - lower_bound) / 2
        
    return ext_data
           
    
def compare(resfolder, algorithms, metric, nvars, plotStyle, plot_type = plotType.LinewErrorBar, bootStrap = False, show = False, xLabel = '# vars'):
   
    toPlot = {}
    for algo in algorithms:
        if algo == Algo.CAnDOIT:
            toPlot[f"{algo.value}_best"] = {"samples" : list(), "means" : list(), "confidences" : list()}
            toPlot[f"{algo.value}_mean"] = {"samples" : list(), "means" : list(), "confidences" : list()}
        else:
            toPlot[algo.value] = {"samples" : list(), "means" : list(), "confidences" : list()} 
        
    for n in range(nvars[0],nvars[1]+1):
        res_path = os.getcwd() + "/results/" + resfolder + "/" + str(n) + ".json"
        
        for algo in toPlot.keys():
            if algo.startswith('candoit'):
                mode = algo.split('_')[1]
                ext_data = extract_data(res_path, 'candoit', metric, mode = ExtractDataMode.MeandStd if not bootStrap else ExtractDataMode.BootStrap, candoit_mode=mode)
                toPlot[algo]["samples"].append(ext_data['candoit']["samples"])
                toPlot[algo]["means"].append(ext_data['candoit']["mean"])
                toPlot[algo]["confidences"].append(ext_data['candoit']["confidence"])
            else:
                ext_data = extract_data(res_path, algo, metric, mode = ExtractDataMode.MeandStd if not bootStrap else ExtractDataMode.BootStrap)
                toPlot[algo]["samples"].append(ext_data[algo]["samples"])
                toPlot[algo]["means"].append(ext_data[algo]["mean"])
                toPlot[algo]["confidences"].append(ext_data[algo]["confidence"])
            # toPlot[algo.value]["samples"].append(ext_data[algo.value]["samples"])
            # toPlot[algo.value]["means"].append(ext_data[algo.value]["mean"])
            # toPlot[algo.value]["confidences"].append(ext_data[algo.value]["confidence"])
            
    if plot_type != plotType.BoxPlot:
        fig, ax = plt.subplots(figsize=(6,4))

        if plot_type == plotType.Line:
            for algo in toPlot.keys():
                plt.plot(range(nvars[0], nvars[1]+1), toPlot[algo]["means"], 
                        marker=plotStyle[algo]['marker'], color = plotStyle[algo]['color'], linestyle = plotStyle[algo]['linestyle'])
            
        elif plot_type == plotType.LinewErrorBar:
            for algo in toPlot.keys():
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
        plt.ylabel(plotLabel[metric])
        bbox_to_anchor = (0, 1.05, 1, .105)
        # bbox_to_anchor = (-0.1, 1.05, 1.2, .105)
        plt.legend([plotLabel[algo] for algo in toPlot.keys()], loc=9, bbox_to_anchor=bbox_to_anchor, ncol=7, mode='expand',)
        plt.grid()
        # plt.title(titleLabel[metric] + ' comparison')
        
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
        ax1.set_ylabel(plotLabel[metric])

        ax1.grid()
        ax1.legend([bp['boxes'][0] for bp in boxplots], [plotLabel[algo] for algo in algorithms], loc='best')
         
    if show:
        plt.show()
    else:
        plt.savefig(os.getcwd() + "/results/" + resfolder + "/" + metric.value + '.pdf')
        plt.savefig(os.getcwd() + "/results/" + resfolder + "/" + metric.value + '.png')


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
    
    # To use to plot RS_comparison_nconfounded
    resfolder = 'AIS_major/AIS_major_S2'
    vars = [5, 7]
    
    
    bootstrap = True
    algorithms = [Algo.CAnDOIT, Algo.LPCMCI]
    plot_style = {'candoit_best': {"marker" : 'o', "color" : 'b', "linestyle" : '-'},
                  'candoit_mean': {"marker" : '^', "color" : 'g', "linestyle" : ':'},
                  Algo.LPCMCI.value: {"marker": 'x', "color": 'r', "linestyle": '-.'}
                  }
    metrics = [Metric.TIME, 
               Metric.ADJ_F1SCORE, Metric.ADJ_PREC, Metric.ADJ_RECA, Metric.ADJ_SHD, Metric.ADJ_FPR,
               Metric.GRAPH_F1SCORE, Metric.GRAPH_PREC, Metric.GRAPH_RECA, Metric.GRAPH_SHD, Metric.GRAPH_FPR,
               Metric.UNCERTAINTY, Metric.PAGSIZE]
    for metric in metrics:
        # compare(r, algorithms, metric, vars, plot_style, plotType.LinewErrorBar, bootStrap = bootstrap, xLabel = '# confounded vars')
        compare(resfolder, algorithms, metric, vars, plot_style, plotType.LinewErrorBar, bootStrap = bootstrap)