"""This module provides methods for computing various metrics."""

from copy import deepcopy

def fully_connected_dag(features, min_lag, max_lag, alsoOrient = False) -> dict:
    """
    Build a fully connected DAG.

    Args:
        features (list): variable list
        min_lag (int): min time lag.
        max_lag (int): max time lag.
        alsoOrient (bool, optional): Flag to compute the metric considering also the orientation of the link and not only its presence. Default False.

    Returns:
        dict: fully connected dag
    """
    if not alsoOrient:
        g = {f: list() for f in features}
        for t in g:
            for s in g:
                for l in range(min_lag, max_lag + 1):
                    if s == t and l == 0: continue 
                    g[t].append((s, -abs(l)))
    else:
        g = {f: {} for f in features}
        for t in g:
            for s in g:
                for l in range(min_lag, max_lag + 1):
                    if s == t and l == 0: continue 
                    g[t][(s, -abs(l))] = ['-->', 'o->', '<->', 'o-o']
    return g


def get_TP(gt, cm, alsoOrient = False) -> int:
    """
    Compute true positive number: edge present in the causal model and present in the groundtruth.

    Args:
        gt (dict): ground-truth SCM.
        cm (dict): estimated SCM.
        alsoOrient (bool, optional): Flag to compute the metric considering also the orientation of the link and not only its presence. Default False.

    Returns:
        int: true positive.
    """
    counter = 0
    if not alsoOrient:
        for t in cm.keys():
            for s in cm[t]:
                if s in gt[t]: counter += 1
    else:
        for t in cm.keys():
            for s in cm[t]:
                if s in gt[t]:
                    if gt[t][s][0] == cm[t][s][0]: counter += 1
                    if gt[t][s][1] == cm[t][s][1]: counter += 1
                    if gt[t][s][2] == cm[t][s][2]: counter += 1
    return counter

 
def get_TN(gt, min_lag, max_lag, cm, alsoOrient = False) -> int:
    """
    Compute true negative number: edge absent in the groundtruth and absent in the causal model.
    
    Args:
        gt (dict): ground-truth SCM.
        min_lag (int): min time lag.
        max_lag (int): max time lag.
        cm (dict): estimated SCM.
        alsoOrient (bool, optional): Flag to compute the metric considering also the orientation of the link and not only its presence. Default False.
        
    Returns:
        int: true negative.
    """
    fullg = fully_connected_dag(list(gt.keys()), min_lag, max_lag, alsoOrient)
    counter = 0
    if not alsoOrient:
        gt_TN = deepcopy(fullg)
        
        # Build the True Negative graph [complementary graph of the ground-truth]
        for t in fullg:
            for s in fullg[t]:
                if s in gt[t]:
                    gt_TN[t].remove(s)
                    
        for t in gt_TN.keys():
            for s in gt_TN[t]:
                if s not in cm[t]: counter += 1
    else:
        gt_TN = deepcopy(fullg)
        
        # Build the True Negative graph [complementary graph of the ground-truth]
        for t in fullg:
            for s in fullg[t]:
                for edge in fullg[t][s]:
                    if s not in gt[t]: continue
                    if edge == gt[t][s]:
                        gt_TN[t][s].remove(edge)
                if len(fullg[t][s]) == 0: del gt_TN[t][s]
                    
        for t in gt_TN.keys():
            for s in gt_TN[t]:
                if s not in cm[t]: 
                    counter += len(gt_TN[t][s])
                else:
                    for edge in gt_TN[t][s]:
                        if edge != cm[t][s]: counter += 1
        
    return counter
    
     
def get_FP(gt, cm, alsoOrient = False) -> int:
    """
    Compute false positive number: edge present in the causal model but absent in the groundtruth.
    
    Args:
        gt (dict): ground-truth SCM.
        cm (dict): estimated SCM.
        alsoOrient (bool, optional): Flag to compute the metric considering also the orientation of the link and not only its presence. Default False.
        
    Returns:
        int: false positive.
    """
    counter = 0
    if not alsoOrient:
        for t in cm.keys():
            for s in cm[t]:
                if s not in gt[t]: counter += 1
    else:
        for t in cm.keys():
            for s in cm[t]:
                if s not in gt[t]: counter += 3
                else:
                    if gt[t][s][0] != cm[t][s][0]: counter += 1
                    if gt[t][s][1] != cm[t][s][1]: counter += 1
                    if gt[t][s][2] != cm[t][s][2]: counter += 1
    return counter


def get_FN(gt, cm, alsoOrient = False) -> int:
    """
    Compute false negative number: edge present in the groundtruth but absent in the causal model.
        
    Args:
        gt (dict): ground-truth SCM.
        cm (dict): estimated SCM.
        alsoOrient (bool, optional): Flag to compute the metric considering also the orientation of the link and not only its presence. Default False.
        
    Returns:
        int: false negative.
    """
    counter = 0
    if not alsoOrient:
        for t in gt.keys():
            for s in gt[t]:
                if s not in cm[t]: counter += 1
    else:
        for t in gt.keys():
            for s in gt[t]:
                if s not in cm[t]: counter += 3
                else:
                    if gt[t][s][0] != cm[t][s][0]: counter += 1
                    if gt[t][s][1] != cm[t][s][1]: counter += 1
                    if gt[t][s][2] != cm[t][s][2]: counter += 1
    return counter
    
    
def shd(gt, cm, alsoOrient = False) -> int:
    """
    Compute Structural Hamming Distance between ground-truth causal graph and the estimated one.

    Args:
        gt (dict): ground-truth SCM.
        cm (dict): estimated SCM.
        alsoOrient (bool, optional): Flag to compute the metric considering also the orientation of the link and not only its presence. Default False.

    Returns:
        int: shd.
    """
    fn = get_FN(gt, cm, alsoOrient)
    fp = get_FP(gt, cm, alsoOrient)
    return fn + fp


def precision(gt, cm, alsoOrient = False) -> float:
    """
    Compute Precision between ground-truth causal graph and the estimated one.

    Args:
        gt (dict): ground-truth SCM.
        cm (dict): estimated SCM.
        alsoOrient (bool, optional): Flag to compute the metric considering also the orientation of the link and not only its presence. Default False.

    Returns:
        float: precision.
    """
    tp = get_TP(gt, cm, alsoOrient)
    fp = get_FP(gt, cm, alsoOrient)
    if tp + fp == 0: return 0
    return tp/(tp + fp)

        
def recall(gt, cm, alsoOrient = False) -> float:
    """
    Compute Recall between ground-truth causal graph and the estimated one.

    Args:
        gt (dict): ground-truth SCM.
        cm (dict): estimated SCM.
        alsoOrient (bool, optional): Flag to compute the metric considering also the orientation of the link and not only its presence. Default False.

    Returns:
        float: recall.
    """
    tp = get_TP(gt, cm, alsoOrient)
    fn = get_FN(gt, cm, alsoOrient)
    if tp + fn == 0: return 0
    return tp/(tp + fn)


def f1_score(gt, cm, alsoOrient = False) -> float:
    """
    Compute F1-score between ground-truth causal graph and the estimated one.

    Args:
        gt (dict): ground-truth SCM.
        cm (dict): estimated SCM.
        alsoOrient (bool, optional): Flag to compute the metric considering also the orientation of the link and not only its presence. Default False.

    Returns:
        float: F1-score.
    """
    p = precision(gt, cm, alsoOrient)
    r = recall(gt, cm, alsoOrient)
    if p + r == 0: return 0
    return (2 * p * r) / (p + r)
    
     
def FPR(gt, min_lag, max_lag, cm, alsoOrient = False) -> float:
    """
    Compute False Positve Rate between ground-truth causal graph and the estimated one.

    Args:
        gt (dict): ground-truth SCM.
        min_lag (int): min time lag.
        max_lag (int): max time lag.
        cm (dict): estimated SCM.
        alsoOrient (bool, optional): Flag to compute the metric considering also the orientation of the link and not only its presence. Default False.

    Returns:
        float: false positive rate.
    """
    fp = get_FP(gt, cm, alsoOrient)
    tn = get_TN(gt, min_lag, max_lag, cm, alsoOrient)
    if tn + fp == 0: return 0
    return fp / (tn + fp)
    
    
def TPR(gt, cm, alsoOrient = False) -> float:
    """
    Compute True Positive Rate between ground-truth causal graph and the estimated one.

    Args:
        gt (dict): ground-truth SCM.
        cm (dict): estimated SCM.
        alsoOrient (bool, optional): Flag to compute the metric considering also the orientation of the link and not only its presence. Default False.

    Returns:
        float: true positive rate.
    """
    tp = get_TP(gt, cm, alsoOrient)
    fn = get_FN(gt, cm, alsoOrient)
    if tp + fn == 0: return 0
    return tp / (tp + fn)


def TNR(gt, min_lag, max_lag, cm, alsoOrient = False) -> float:
    """
    Compute True Negative Rate between ground-truth causal graph and the estimated one.

    Args:
        gt (dict): ground-truth SCM.
        min_lag (int): min time lag.
        max_lag (int): max time lag.
        cm (dict): estimated SCM.
        alsoOrient (bool, optional): Flag to compute the metric considering also the orientation of the link and not only its presence. Default False.

    Returns:
        float: true negative rate.
    """
    tn = get_TN(gt, min_lag, max_lag, cm, alsoOrient)
    fp = get_FP(gt, cm, alsoOrient)
    if tn + fp == 0: return 0
    return tn / (tn + fp)


def FNR(gt, cm, alsoOrient = False) -> float:
    """
    Compute False Negative Rate between ground-truth causal graph and the estimated one.

    Args:
        gt (dict): ground-truth SCM.
        cm (dict): estimated SCM.
        alsoOrient (bool, optional): Flag to compute the metric considering also the orientation of the link and not only its presence. Default False.

    Returns:
        float: false negative rate.
    """
    fn = get_FN(gt, cm, alsoOrient)
    tp = get_TP(gt, cm, alsoOrient)
    if tp + fn == 0: return 0
    return fn / (tp + fn)