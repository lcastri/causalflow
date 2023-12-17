from abc import ABC, abstractmethod
import sys
import copy
import pickle
from connectingdots.graph.DAG import DAG
from connectingdots.CPrinter import CPLevel, CP
from connectingdots.basics.constants import *
from connectingdots.basics.logger import Logger
import connectingdots.basics.utils as utils
from connectingdots.preprocessing.data import Data


class CausalDiscoveryMethod(ABC):
    """
    CausalDiscoveryMethod class.

    CausalDiscoveryMethod is an abstract causal discovery method for 
    large-scale time series datasets.
    """

    def __init__(self, 
                 data: Data, 
                 min_lag, max_lag, 
                 verbosity: CPLevel, 
                 alpha = 0.05, 
                 resfolder = None,
                 neglect_only_autodep = False):
        """
        CausalDiscoveryMethod class contructor

        Args:
            data (Data): data to analyse
            min_lag (int): minimum time lag
            max_lag (int): maximum time lag
            verbosity (CPLevel): verbosity level
            alpha (float, optional): significance level. Defaults to 0.05.
            resfolder (string, optional): result folder to create. Defaults to None.
            neglect_only_autodep (bool, optional): Bit for neglecting variables with only autodependency. Defaults to False.

        """
        
        self.data = data
        self.alpha = alpha
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.CM = DAG(self.data.features, min_lag, max_lag, neglect_only_autodep)
        self.neglect_only_autodep = neglect_only_autodep

        self.respath, self.dag_path, self.ts_dag_path = None, None, None
        if resfolder is not None:
            utils.create_results_folder()
            logpath, self.respath, self.dag_path, self.ts_dag_path = utils.get_selectorpath(resfolder)  
            sys.stdout = Logger(logpath)
        
        CP.set_verbosity(verbosity)


    @abstractmethod
    def run(self) -> DAG:
        """
        Run causal discovery method
        
        Returns:
            DAG: causal model
        """
        pass
    

    def dag(self,
            node_layout = 'dot',
            min_width = 1,
            max_width = 5,
            min_score = 0,
            max_score = 1,
            node_size = 8,
            node_color = 'orange',
            edge_color = 'grey',
            font_size = 12,
            label_type = LabelType.Lag,
            save_name = None,
            img_ext = ImageExt.PNG):
        """
        Saves dag plot if resfolder has been set otherwise it shows the figure
        
        Args:
            node_layout (str, optional): Node layout. Defaults to 'dot'.
            min_width (int, optional): minimum linewidth. Defaults to 1.
            max_width (int, optional): maximum linewidth. Defaults to 5.
            min_score (int, optional): minimum score range. Defaults to 0.
            max_score (int, optional): maximum score range. Defaults to 1.
            node_size (int, optional): node size. Defaults to 8.
            node_color (str, optional): node color. Defaults to 'orange'.
            edge_color (str, optional): edge color. Defaults to 'grey'.
            font_size (int, optional): font size. Defaults to 12.
            label_type (LabelType, optional): enum to set whether to show the lag time (LabelType.Lag) or the strength (LabelType.Score) of the dependencies on each link/node or not showing the labels (LabelType.NoLabels). Default LabelType.Lag.
            img_ext (ImageExt, optional): dag image extention (.png, .pdf, ..). Default ImageExt.PNG.
        """
        
        if self.CM:
            if save_name is None: save_name = self.dag_path
            try:
                self.CM.dag(node_layout, min_width, 
                            max_width, min_score, max_score,
                            node_size, node_color, edge_color,
                            font_size, label_type, save_name,
                            img_ext)
            except:
                CP.warning("node_layout = " + node_layout + " generates error. node_layout = circular used.")
                self.CM.dag("circular", min_width, 
                            max_width, min_score, max_score,
                            node_size, node_color, edge_color,
                            font_size, label_type, save_name,
                            img_ext)
        else:
            CP.warning("Dag impossible to create: causal model not estimated yet")
    
        
    def timeseries_dag(self,
                       min_width = 1,
                       max_width = 5,
                       min_score = 0,
                       max_score = 1,
                       node_size = 8,
                       font_size = 12,
                       node_color = 'orange',
                       edge_color = 'grey',
                       save_name = None,
                       img_ext = ImageExt.PNG):
        """
        Saves timeseries dag plot if resfolder has been set otherwise it shows the figure
        
        Args:
            min_width (int, optional): minimum linewidth. Defaults to 1.
            max_width (int, optional): maximum linewidth. Defaults to 5.
            min_score (int, optional): minimum score range. Defaults to 0.
            max_score (int, optional): maximum score range. Defaults to 1.
            node_size (int, optional): node size. Defaults to 8.
            node_color (str, optional): node color. Defaults to 'orange'.
            edge_color (str, optional): edge color. Defaults to 'grey'.
            font_size (int, optional): font size. Defaults to 12.
            img_ext (ImageExt, optional): dag image extention (.png, .pdf, ..). Default ImageExt.PNG.
        """
        
        if self.CM:
            if save_name is None: save_name = self.ts_dag_path
            self.CM.ts_dag(self.max_lag, min_width,
                           max_width, min_score, max_score,
                           node_size, node_color, edge_color,
                           font_size, save_name, img_ext)
        else:
            CP.warning("Timeseries dag impossible to create: causal model not estimated yet")
            
    
    def load(self, res_path):
        """
        Loads previously estimated result 

        Args:
            res_path (str): pickle file path
        """
        with open(res_path, 'rb') as f:
            r = pickle.load(f)
            self.CM = r['causal_model']
            self.alpha = r['alpha']
            self.dag_path = r['dag_path']
            self.ts_dag_path = r['ts_dag_path']
            
            
    def save(self):
        """
        Save causal discovery result as pickle file if resfolder is set
        """
        if self.respath is not None:
            if self.CM:
                res = dict()
                res['causal_model'] = copy.deepcopy(self.CM)
                res['alpha'] = self.alpha
                res['dag_path'] = self.dag_path
                res['ts_dag_path'] = self.ts_dag_path
                with open(self.respath, 'wb') as resfile:
                    pickle.dump(res, resfile)
            else:
                CP.warning("Causal model impossible to save")