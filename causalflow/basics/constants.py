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


ADJ = 'adj'
P_Y_GIVEN_DOX_ADJ = 'p_y|do(x)_adj'
P_Y_GIVEN_DOX = 'p_y|do_x'


class LabelType(Enum):
    Lag = "Lag"
    Score = "Score"
    NoLabels = "NoLabels"
    
    
class LinkType(Enum):
    Directed = "-->"
    Undirected = "o-o"
    
    
class ImageExt(Enum):
    PNG = ".png"
    PDF = ".pdf"
    JPG = ".jpg"