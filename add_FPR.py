import copy
import json
import os

from fpcmci.graph.DAG import DAG

_TIME = 'time'
_PREC = 'precision'
_RECA = 'recall'
_F1SCORE = 'f1_score'
_SHD = "shd"
_FN = "fn"
_FP = "fp"
_TP = "tp"
_FPR = "fpr"
_FPCMCI = 'fpcmci'
_doFPCMCI = 'dofpcmci'
_SCM = 'scm'    
    
    
def FPR(gt, cm):
    """
    Computes False Positve Rate between ground-truth causal graph and the estimated one

    Args:
        cm (dict): estimated SCM

    Returns:
        float: false positive rate
    """
    fp = get_FP(gt, cm)
    tn = get_TN(gt, cm)
    if tn + fp == 0: return 0
    return fp / (tn + fp)
    
    
def get_FP(gt, cm):
    """
    False positive number:
    edge present in the causal model 
    but absent in the groundtruth

    Args:
        cm (dict): estimated SCM

    Returns:
        int: false positive
    """
    counter = 0
    for node in cm.keys():
        for edge in cm[node]:
            if edge not in gt[node]: counter += 1
    return counter


def get_TN(gt, cm):
    """
    True negative number:
    edge absent in the groundtruth 
    and absent in the causal model
    
    Args:
        cm (dict): estimated SCM

    Returns:
        int: true negative
    """
    fullg = DAG(list(gt.keys()), 1, 2, None)
    fullg.fully_connected_dag()
    fullscm = fullg.get_SCM()
    gt_TN = copy.deepcopy(fullscm)
    
    # Build the True Negative graph [complementary graph of the ground-truth]
    for node in fullscm:
        for edge in fullscm[node]:
            if edge in gt[node]:
                gt_TN[node].remove(edge)
                
    counter = 0
    for node in gt_TN.keys():
        for edge in gt_TN[node]:
            if edge not in cm[node]: counter += 1
    return counter
    

def addFalsePositiveRate(resfolder, nvars):
    for n in range(nvars[0], nvars[1]+1):
        res_path = os.getcwd() + "/results/" + resfolder + "/" + str(n) + ".json"
        with open(res_path) as json_file:
            r = json.load(json_file)
            for i in r.keys():
                GT = eval(r[i]["GT"])
                fpcmci_scm = eval(r[i][_FPCMCI][_SCM])
                dofpcmci_scm = eval(r[i][_doFPCMCI][_SCM])
                r[i][_FPCMCI][_FPR] = FPR(GT, fpcmci_scm)
                r[i][_doFPCMCI][_FPR] = FPR(GT, dofpcmci_scm)
                
        # Save the dictionary back to a JSON file
        with open(res_path, 'w') as file:
            json.dump(r, file)

    
if __name__ == '__main__':   

    resfolder = ['single_conf_nonlin_1000_1000_0_0.5']
    for r in resfolder:
        addFalsePositiveRate(r, [7, 14])