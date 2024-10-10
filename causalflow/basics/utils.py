"""This module provides utilities methods."""

import copy
import os
from pathlib import Path
from causalflow.basics.constants import *


def cls():
    """Clear terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_selectorpath(resfolder):
    """
    Return log file path.

    Args:
        resfolder (str): result folder.

    Returns:
        (str): log file path.
    """
    Path(resfolder).mkdir(parents=True, exist_ok=True)
    return SEP.join([resfolder, LOG_FILENAME]), SEP.join([resfolder, RES_FILENAME]), SEP.join([resfolder, DAG_FILENAME]), SEP.join([resfolder, TSDAG_FILENAME])


def create_results_folder(resfolder):
    """
    Create results folder if doesn't exist.

    Args:
        resfolder (str): result folder's name.
    """
    Path(resfolder).mkdir(parents=True, exist_ok=True)
    
    
def remove_from_list(list: list(), item) -> list():
    """
    Create a copy of a list and remove from it an item.

    Args:
        list (list): list.
        item (any): item to remove.

    Returns:
        list: new list without the item.
    """
    tmp = copy.deepcopy(list)
    tmp.remove(item)
    return tmp
