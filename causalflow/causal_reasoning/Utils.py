import numpy as np

def expectation(y, p):
    """
    Compute the expectation E[y*p(y)/sum(p(y))].

    Args:
        y (ndarray): process samples.
        p (ndarray): probability density function. Note it must be ALREADY NORMALISED.

    Returns:
        float: expectation E[y*p(y)/sum(p(y))].
    """
    if np.sum(p) == 0:
        return np.nan
    expectation_Y_given_X = np.sum(y * p)
    return expectation_Y_given_X
    
    
def mode(y, p):
    """
    Compute the mode, which is the most likely valueof y.

    Args:
        y (ndarray): process samples.
        p (ndarray): probability density function. Note it must be ALREADY NORMALISED.
        
    Returns:
        float: mode Mode(y*p(y)).
    """
    return y[np.argmax(p)]

    
def normalise(p):
    """
    Normalise the probability density function to ensure it sums to 1.

    Args:
        p (ndarray): probability density function.

    Returns:
        ndarray: normalised probability density function.
    """
    p_sum = np.sum(p)
    if p_sum > 0:
        return p / p_sum
    else:
        return np.zeros_like(p)
    
    
def format_combo(combo):
    return tuple(sorted(combo))