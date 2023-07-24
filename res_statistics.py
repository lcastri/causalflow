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
_SCM = 'scm'
_GT = 'gt'

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
    
    
def save_result(r, startTime, stopTime, scm, alg, gt):
    r[alg][_GT] = str(gt)
    r[alg][_SCM] = str(scm)
    r[alg][_TIME] = str(stopTime - startTime)
    r[alg][_PREC] = precision(gt, cm = scm)
    r[alg][_RECA] = recall(gt, cm = scm)
    r[alg][_F1SCORE] = f1_score(r[alg][_PREC], r[alg][_RECA])
    r[alg][_SHD] = shd(gt, cm = scm)
    print(alg + " statistics -- |time = " + str(r[alg][_TIME]) + " -- |F1 score = " + str(r[alg][_F1SCORE]) + " -- |SHD = " + str(r[alg][_SHD]))
    return r



def get_TP(gt, cm):
    """
    True positive rate:
    edge present in the causal model 
    and present in the groundtruth

    Args:
        gt (dict): groundtruth
        cm (dict): causal model

    Returns:
        int: true positive
    """
    counter = 0
    for node in cm.keys():
        for edge in cm[node]:
            if edge in gt[node]: counter += 1
    return counter


def get_FP(gt, cm):
    """
    False positive rate:
    edge present in the causal model 
    but absent in the groundtruth

    Args:
        gt (dict): groundtruth
        cm (dict): causal model

    Returns:
        int: false positive
    """
    counter = 0
    for node in cm.keys():
        for edge in cm[node]:
            if edge not in gt[node]: counter += 1
    return counter


def get_FN(gt, cm):
    """
    False negative rate:
    edge present in the groundtruth 
    but absent in the causal model
    
    Args:
        gt (dict): groundtruth
        cm (dict): causal model

    Returns:
        int: false negative
    """
    counter = 0
    for node in gt.keys():
        for edge in gt[node]:
            if edge not in cm[node]: counter += 1
    return counter


def shd(gt, cm):
    fn = get_FN(gt, cm)
    fp = get_FP(gt, cm)
    return fn + fp


def precision(gt, cm):
    tp = get_TP(gt, cm)
    fp = get_FP(gt, cm)
    if tp + fp == 0: return 0
    return tp/(tp + fp)

    
def recall(gt, cm):
    tp = get_TP(gt, cm)
    fn = get_FN(gt, cm)
    if tp + fn == 0: return 0
    return tp/(tp + fn)


def f1_score(p, r):
    if p + r == 0: return 0
    return (2 * p * r) / (p + r)
    
    
def collect_data(file_path, data):
    since = datetime.datetime(1900, 1, 1, 0, 0, 0, 0)
    with open(file_path) as json_file:
        r = json.load(json_file)
        data_fpcmci = list()
        data_dofpcmci = list()
        for i in r.keys():
            if data == _TIME:
                time_tepcmci = datetime.datetime.strptime(r[i][_FPCMCI][data], '%H:%M:%S.%f')
                time_pcmci = datetime.datetime.strptime(r[i][_doFPCMCI][data], '%H:%M:%S.%f')
                data_fpcmci.append((time_tepcmci - since).total_seconds())
                data_dofpcmci.append((time_pcmci - since).total_seconds())
            else:
                data_fpcmci.append(r[i][_FPCMCI][data])
                data_dofpcmci.append(r[i][_doFPCMCI][data])

    return sum(data_fpcmci)/len(r.keys()), np.std(data_fpcmci), sum(data_dofpcmci)/len(r.keys()), np.std(data_dofpcmci)
        
        
    
def plot_statistics(resfolder, data, nvars, plot_type = plotType.ErrorBar):
    fpcmci_means = list()
    fpcmci_stds = list()
    dofpcmci_means = list()
    dofpcmci_stds = list()
    
    for n in range(nvars[0],nvars[1]+1):
        res_path = os.getcwd() + "/results/" + resfolder + "/" + str(n) + ".json"
        fpcmci_mean, fpcmci_std, dofpcmci_mean, dofpcmci_std = collect_data(res_path, data)
        fpcmci_means.append(fpcmci_mean)
        fpcmci_stds.append(fpcmci_std)
        dofpcmci_means.append(dofpcmci_mean)
        dofpcmci_stds.append(dofpcmci_std)
        
    fig, ax = plt.subplots(figsize=(6,4))
    
    if plot_type == plotType.Normal:
        plt.plot(range(nvars[0], nvars[1]+1), dofpcmci_means)
        plt.plot(range(nvars[0], nvars[1]+1), fpcmci_means)
    elif plot_type == plotType.ErrorBar:
        plt.errorbar(range(nvars[0], nvars[1]+1), dofpcmci_means, dofpcmci_stds, marker='o', capsize = 5, color = 'b')
        plt.errorbar(range(nvars[0], nvars[1]+1), fpcmci_means, fpcmci_stds, marker='^', capsize = 5, color = 'r', linestyle = '--')
    elif plot_type == plotType.ErrorBand:
        plt.plot(range(nvars[0], nvars[1]+1), dofpcmci_means, marker='o', color = 'b')
        plt.plot(range(nvars[0], nvars[1]+1), fpcmci_means, marker='^', color = 'r', linestyle = '--')
        plt.fill_between(range(nvars[0], nvars[1]+1), np.array(dofpcmci_means) - np.array(dofpcmci_stds), np.array(dofpcmci_means) + np.array(dofpcmci_stds), alpha=0.3, color = 'b')
        plt.fill_between(range(nvars[0], nvars[1]+1), np.array(fpcmci_means) - np.array(fpcmci_stds), np.array(fpcmci_means) + np.array(fpcmci_stds), alpha=0.3, color = 'r')
    
    plt.xticks(range(nvars[0],nvars[1]+1))
    if data is _PREC or data is _RECA or data is _F1SCORE: plt.ylim(0, 1.1)
    plt.xlabel("# vars")
    plt.ylabel(dlabel[data])
    plt.legend([dlabel[_doFPCMCI], dlabel[_FPCMCI]])
    plt.grid()
    plt.title(data + ' comparison')
    plt.savefig(os.getcwd() + "/results/" + resfolder + "/" + data + '.pdf')
    plt.savefig(os.getcwd() + "/results/" + resfolder + "/" + data + '.png')

    
if __name__ == '__main__':   
    resfolder = 'My_RS_FPCMCI_vs_doFPCMCI_lin'

    plot_statistics(resfolder, _TIME, [4, 10], plotType.ErrorBar)
    plot_statistics(resfolder, _F1SCORE, [4, 10], plotType.ErrorBar)
    plot_statistics(resfolder, _PREC, [4, 10], plotType.ErrorBar)
    plot_statistics(resfolder, _RECA, [4, 10], plotType.ErrorBar)
    plot_statistics(resfolder, _SHD, [4, 10], plotType.ErrorBar)