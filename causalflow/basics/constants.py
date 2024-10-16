"""
This module provides the support classes and various constants.

Classes:
    LabelType: label type.
    LinkType: link type.
    ImageExt: image extention.
"""

from enum import Enum

SOURCE = 'source'
SCORE = 'score'
PVAL = 'pval'
LAG = 'lag'
TYPE = 'type'

DASH = '-' * 55
SEP = "/"
RES_FILENAME = "res.pkl"
DAG_FILENAME = "dag"
TSDAG_FILENAME = "ts_dag"
LOG_FILENAME = "log.txt"


class LabelType(Enum):
    """LabelType Enumerator."""
    
    Lag = "Lag"
    Score = "Score"
    NoLabels = "NoLabels"
    
    
class DataType(Enum):
    """DataType Enumerator."""
    
    Continuos = 0
    Discrete = 1
    
    
class LinkType(Enum):
    """LinkType Enumerator."""
    
    Directed = "-->"
    Uncertain = "o-o"
    Bidirected = "<->"
    HalfUncertain = "o->"
    
    
class ImageExt(Enum):
    """ImageExt Enumerator."""
    
    PNG = ".png"
    PDF = ".pdf"
    JPG = ".jpg"