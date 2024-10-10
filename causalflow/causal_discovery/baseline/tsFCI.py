"""
This module provides the tsFCI class.

Classes:
    tsFCI: class containing the tsFCI causal discovery algorithm.
"""

from pathlib import Path
from subprocess import Popen, PIPE
import os
import pandas as pd
from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
from causalflow.causal_discovery.CausalDiscoveryMethod import CausalDiscoveryMethod 
from causalflow.causal_discovery.baseline.pkgs import utils

class tsFCI(CausalDiscoveryMethod):
    """tsFCI causal discovery method."""
    
    def __init__(self, 
                 data, 
                 min_lag,
                 max_lag, 
                 verbosity, 
                 alpha = 0.05, 
                 resfolder = None,
                 neglect_only_autodep = False,
                 clean_cls = True):
        """
        Class constructor.

        Args:
            data (Data): data to analyse.
            min_lag (int): minimum time lag.
            max_lag (int): maximum time lag.
            verbosity (CPLevel): verbosity level.
            alpha (float, optional): PCMCI significance level. Defaults to 0.05.
            resfolder (string, optional): result folder to create. Defaults to None.
            neglect_only_autodep (bool, optional): Bit for neglecting variables with only autodependency. Defaults to False.
            clean_cls (bool): Clean console bit. Default to True.
        """
        super().__init__(data, min_lag, max_lag, verbosity, alpha, resfolder, neglect_only_autodep, clean_cls)
                
    
    def run(self) -> DAG:
        """
        Run causal discovery algorithm.

        Returns:
            (DAG): estimated causal model.
        """
        # Remove all arguments from directory
        dir_path = os.path.dirname(os.path.realpath(__file__))
        Path(dir_path + "/args").mkdir(exist_ok=True)
        Path(dir_path + "/results").mkdir(exist_ok=True)
            
        script = dir_path + "/pkgs/tsfci.R"
        r_arg_list = []
            
        # COMMAND WITH ARGUMENTS
        self.data.d.to_csv(dir_path + "/args/data.csv", index=False)
        r_arg_list.append(dir_path + "/args/data.csv")
        r_arg_list.append(str(self.alpha))
        r_arg_list.append(str(self.max_lag))

        r_arg_list.append(dir_path)
        cmd = ["Rscript", script] + r_arg_list

        p = Popen(cmd, cwd="./", stdin=PIPE, stdout=PIPE, stderr=PIPE)
            
        # Return R output or error
        output, error = p.communicate()
        print(output.decode('utf-8'))
        if p.returncode == 0:
            g_df = pd.read_csv(dir_path + "/results/result.csv", header=0, index_col=0)
            g_dict = self.ts_fci_dataframe_to_dict(g_df, self.data.features, self.max_lag)
            self.CM = self._to_DAG(g_dict)
            utils.clean(dir_path)
            
            if self.resfolder is not None: self.logger.close()
            return self.CM

        else:
            utils.clean(dir_path)
            print('R Error:\n {0}'.format(error.decode('utf-8')))
            exit(0)
           
    
    def _to_DAG(self, graph):
        """
        Re-elaborate the result in a DAG.
        
        Args:
            graph (dict): graph to convert into a DAG

        Returns:
            (DAG): result re-elaborated.
        """
        tmp_dag = DAG(self.data.features, self.min_lag, self.max_lag, self.neglect_only_autodep)
        tmp_dag.sys_context = dict()
        for t in graph.keys():
            for s in graph[t]:
                lag = abs(s[1])
                if lag >= self.min_lag and lag <= self.max_lag:
                    tmp_dag.add_source(t, s[0], utils.DSCORE, 0, s[1])
        return tmp_dag


    def ts_fci_dataframe_to_dict(self, df, names, nlags) -> dict:
        """
        Convert tsFCI result into a dict for _to_DAG.

        Args:
            df (DataFrame): graph.
            names (list[str]): variables' name.
            nlags (int): max time lag.

        Returns:
            dict: dict graph.
        """
        # todo: check if its correct
        for i in range(df.shape[1]):
            for j in range(i+1, df.shape[1]):
                if df[df.columns[i]].loc[df.columns[j]] == 2:
                    if df[df.columns[j]].loc[df.columns[i]] == 2:
                        print(df.columns[i] + " <-> " + df.columns[j])

        g_dict = dict()
        for name_y in names:
            g_dict[name_y] = []
        for ty in range(nlags):
            for name_y in names:
                t_name_y = df.columns[ty*len(names)+names.index(name_y)]
                for tx in range(nlags):
                    for name_x in names:
                        t_name_x = df.columns[tx * len(names) + names.index(name_x)]
                        if df[t_name_y].loc[t_name_x] == 2:
                            if (name_x, tx-ty) not in g_dict[name_y]:
                                g_dict[name_y].append((name_x, tx - ty))
        print(g_dict)
        return g_dict
