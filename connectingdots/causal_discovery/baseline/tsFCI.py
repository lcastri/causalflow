from subprocess import Popen, PIPE
import os
import glob
import pandas as pd


from pathlib import Path
from subprocess import Popen, PIPE
import os
import glob
import pandas as pd
import json      
from connectingdots.graph.DAG import DAG
from connectingdots.preprocessing.data import Data
from connectingdots.causal_discovery.CausalDiscoveryMethod import CausalDiscoveryMethod 
from connectingdots.causal_discovery.baseline.pkgs import utils

class tsFCI(CausalDiscoveryMethod):
    """
    tsFCI causal discovery method.
    """
    def __init__(self, 
                 data, 
                 min_lag,
                 max_lag, 
                 verbosity, 
                 alpha = 0.05, 
                 resfolder = None,
                 neglect_only_autodep = False,):
        
        super().__init__(data, min_lag, max_lag, verbosity, alpha, resfolder, neglect_only_autodep)
                
    
    def run(self) -> DAG:
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
            return self.CM

        else:
            utils.clean(dir_path)
            print('R Error:\n {0}'.format(error.decode('utf-8')))
            exit(0)
           
    
    def _to_DAG(self, graph):
        """
        Re-elaborates the result in a DAG

        Returns:
            (DAG): result re-elaborated
        """
        tmp_dag = DAG(self.data.features, self.min_lag, self.max_lag, self.neglect_only_autodep)
        tmp_dag.sys_context = dict()
        for t in graph.keys():
            for s in graph[t]:
                lag = abs(s[1])
                if lag >= self.min_lag and lag <= self.max_lag:
                    tmp_dag.add_source(t, s[0], utils.DSCORE, 0, s[1])
        # tmp_dag.remove_unneeded_features()
        return tmp_dag




    def ts_fci_dataframe_to_dict(self, df, names, nlags):
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