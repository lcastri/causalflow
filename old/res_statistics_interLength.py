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
_FPCMCI = 'fpcmci'
_doFPCMCI = 'dofpcmci'
PERC_STEP = 5

dlabel = {_TIME : 'time [s]',
          _PREC : 'precision',
          _RECA : 'recall',
          _F1SCORE : 'f1_score',
          _SHD : 'SHD',
          _FPCMCI : 'FPCMCI',
          _doFPCMCI : 'doFPCMCI'}


class plotType(Enum):
    Normal = 0
    ErrorBar = 1
    ErrorBand = 2
  
    
def collect_data(file_path, n, data):
    since = datetime.datetime(1900, 1, 1, 0, 0, 0, 0)
    with open(file_path) as json_file:
        r = json.load(json_file)
        data_dofpcmci = list()
        for i in r[n].keys():
            if data == _TIME:
                time_dofpcmci = datetime.datetime.strptime(r[n][i][data], '%H:%M:%S.%f')
                data_dofpcmci.append((time_dofpcmci - since).total_seconds())
            else:
                data_dofpcmci.append(r[n][i][data])

    return sum(data_dofpcmci)/len(r[n].keys()), np.std(data_dofpcmci)
        
        
    
def plot_statistics(resfolder, data, nvars, plot_type = plotType.ErrorBar):
    dofpcmci_means = list()
    dofpcmci_stds = list()
    percentages = [((100 - i)/100, i/100) for i in range(PERC_STEP, 100, PERC_STEP)]

    for n in range(0,nvars+1):
        res_path = os.getcwd() + "/results/" + resfolder + "/" + str(n) + ".json"
        dofpcmci_mean, dofpcmci_std = collect_data(res_path, str(n), data)
        dofpcmci_means.append(dofpcmci_mean)
        dofpcmci_stds.append(dofpcmci_std)
        
    fig, ax = plt.subplots(figsize=(6,4))
    
    if plot_type == plotType.Normal:
        plt.plot(range(0, nvars+1), dofpcmci_means)
    elif plot_type == plotType.ErrorBar:
        plt.errorbar(range(0, nvars+1), dofpcmci_means, dofpcmci_stds, marker='o', capsize = 5, color = 'b')
    elif plot_type == plotType.ErrorBand:
        plt.plot(range(0, nvars+1), dofpcmci_means, marker='o', color = 'b')
        plt.fill_between(range(0, nvars+1), np.array(dofpcmci_means) - np.array(dofpcmci_stds), np.array(dofpcmci_means) + np.array(dofpcmci_stds), alpha=0.3, color = 'b')
    
    plt.xticks(range(0,nvars+1), labels = percentages, rotation = 90, fontsize = 4)
    if data is _PREC or data is _RECA or data is _F1SCORE: plt.ylim(0, 1.1)
    plt.xlabel("(obs-int) rate")
    plt.ylabel(dlabel[data])
    plt.legend([dlabel[_doFPCMCI]])
    plt.grid()
    plt.title(data + ' intervention length analysis')
    plt.savefig(os.getcwd() + "/results/" + resfolder + "/" + data + '.pdf')

    
if __name__ == '__main__':   
    resfolder = 'doFPCMCI_S1_interLength'
    resnumber = 18
    plot_statistics(resfolder, _TIME, resnumber, plotType.ErrorBar)
    plot_statistics(resfolder, _F1SCORE, resnumber, plotType.ErrorBar)
    plot_statistics(resfolder, _PREC, resnumber, plotType.ErrorBar)
    plot_statistics(resfolder, _RECA, resnumber, plotType.ErrorBar)
    plot_statistics(resfolder, _SHD, resnumber, plotType.ErrorBar)