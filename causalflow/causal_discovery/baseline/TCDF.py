"""
This module provides the TCDF class.

Classes:
    TCDF: class containing the TCDF causal discovery algorithm.
"""

from pathlib import Path
from subprocess import Popen, PIPE
import os
import json
from causalflow.graph.DAG import DAG
from causalflow.causal_discovery.CausalDiscoveryMethod import CausalDiscoveryMethod 
from causalflow.CPrinter import CP
from causalflow.causal_discovery.baseline.pkgs import utils


class TCDF(CausalDiscoveryMethod):
    """TCDF causal discovery method."""
    
    def __init__(self, 
                 data, 
                 min_lag,
                 max_lag, 
                 verbosity, 
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
            resfolder (string, optional): result folder to create. Defaults to None.
            neglect_only_autodep (bool, optional): Bit for neglecting variables with only autodependency. Defaults to False.
            clean_cls (bool): Clean console bit. Default to True.
        """
        super().__init__(data, min_lag, max_lag, verbosity, resfolder=resfolder, neglect_only_autodep=neglect_only_autodep, clean_cls=clean_cls)


    def run(self, 
            epochs=1000,  
            kernel_size=4, 
            dilation_coefficient=4, 
            hidden_layers=0, 
            learning_rate=0.01,
            cuda=False) -> DAG:
        """
        Run causal discovery algorithm.

        Returns:
            (DAG): estimated causal model.
        """
        # Remove all arguments from directory
        dir_path = os.path.dirname(os.path.realpath(__file__))
        Path(dir_path+"/args").mkdir(exist_ok=True)
        Path(dir_path+"/results").mkdir(exist_ok=True)
        script = dir_path + "/pkgs/TCDF-master/runTCDF" + ".py"
        r_arg_list = []
        r_arg_list.append("--epochs")
        r_arg_list.append(str(epochs))
        r_arg_list.append("--kernel_size")
        r_arg_list.append(str(kernel_size))
        r_arg_list.append("--dilation_coefficient")
        r_arg_list.append(str(dilation_coefficient))
        r_arg_list.append("--hidden_layers")
        r_arg_list.append(str(hidden_layers))
        r_arg_list.append("--learning_rate")
        r_arg_list.append(str(learning_rate))
        r_arg_list.append("--significance")
        r_arg_list.append(str(0.8))
        self.data.d.to_csv(dir_path + "/args/data.csv", index=False)
        r_arg_list.append("--data")
        r_arg_list.append(dir_path + "/args/data.csv")            
        if cuda: r_arg_list.append("--cuda")
        r_arg_list.append("--path")
        r_arg_list.append(dir_path)
        
        cmd = ["python", script] + r_arg_list
        p = Popen(cmd, cwd="./", stdin=PIPE, stdout=PIPE, stderr=PIPE)
        
        # Return R output or error
        output, error = p.communicate()
        CP.info(output.decode('utf-8'))
        if p.returncode == 0:
            g_dict = json.load(open(dir_path + "/results/tcdf_result.txt"))
            for key in g_dict.keys():
                key_list = []
                for elem in g_dict[key]:
                    key_list.append(tuple(elem))
                g_dict[key] = key_list
            utils.clean(dir_path)
            self.CM = self._to_DAG(g_dict)

            if self.resfolder is not None: self.logger.close()
            return self.CM
        else:
            utils.clean(dir_path)
            CP.warning('Python Error:\n {0}'.format(error))
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