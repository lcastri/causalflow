from copy import deepcopy
import random
from matplotlib import pyplot as plt
import numpy as np
from tigramite.toymodels import structural_causal_processes as toys 
from dgp.src.data_generation_configs import CausalGraphConfig, DataGenerationConfig, FunctionConfig, NoiseConfig, RuntimeConfig
from dgp.src.time_series_generator import TimeSeriesGenerator
from random_system.functions import FUNCTIONS
from fpcmci.preprocessing.data import Data

class RandomSystem:
    def __init__(self, max_lag, min_lag, nfeature, nsamples, complexity = 20, resfolder = None):
        self.N = nfeature
        self.T = nsamples
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.complexity = complexity
        self.data_gen_config = DataGenerationConfig(random_seed = random.randint(1, 100), complexity = complexity, percent_missing = 0.0,
                                                    causal_graph_config = CausalGraphConfig(
                                                        graph_complexity = complexity,
                                                        include_noise = True,
                                                        max_lag = max_lag,
                                                        min_lag = min_lag,
                                                        num_targets = 0,
                                                        num_features = nfeature,
                                                        num_latent = 0,
                                                        prob_edge = 0.3,
                                                        max_parents_per_variable = 1,
                                                        max_target_parents = 2, max_target_children = 0,
                                                        max_feature_parents = nfeature, max_feature_children = nfeature,
                                                        max_latent_parents = 2, max_latent_children = 2,
                                                        allow_latent_direct_target_cause = False,
                                                        allow_target_direct_target_cause = False,
                                                        prob_target_autoregressive = 0.1,
                                                        prob_feature_autoregressive = 0.3,
                                                        prob_latent_autoregressive = 0.2,
                                                        prob_noise_autoregressive = 0.0,
                                                    ),
                                                    function_config = FunctionConfig(
                                                        function_complexity = complexity,
                                                        functions = ['monotonic'], #['linear', 'piecewise_linear', 'monotonic', 'trigonometric']
                                                        prob_functions = [float(1)],
                                                    ),
                                                    noise_config = NoiseConfig(
                                                        noise_complexity = complexity,
                                                        noise_variance = 0.1,
                                                        distributions = ['uniform'], #['uniform', 'gaussian']
                                                        prob_distributions = [float(1)],
                                                    ),
                                                    runtime_config=RuntimeConfig(
                                                        num_samples = nsamples, 
                                                        data_generating_seed = random.randint(1,100)
                                                    )
                                                    )
        
        # Instantiate a time series generator.
        self.ts_generator = TimeSeriesGenerator(config = self.data_gen_config)
        self.links = None
        self.resfolder = resfolder
        
        
    def randSCM(self):        
        # Generate data sets from this configuration.
        _, cm = self.ts_generator.generate_datasets()
            
        # Causal Model
        self.varlist = list(dict.fromkeys([str(n).split('_')[0] for n in cm.causal_graph.nodes if not n[0]=='S']))
        self.gt = {v : list() for v in self.varlist}
        edge_list = [edge for edge in cm.causal_graph.edges if not edge[0].startswith('S')]
        for edge in edge_list:
            if edge[1][-1] == 't':
                source = edge[0].split("_")[0]
                target = edge[1].split("_")[0]
                lag = edge[0][-1]
                self.gt[target].append((source, -int(lag)))
                
        # View causal graph.
        cm.display_graph(include_noise=False)  # Set to True to graph noise nodes.

        # Show plots.
        if self.resfolder is not None:
            plt.savefig('gt.png')
        else:
            plt.show()
        
        return self.gt, self.varlist
    
    
    def get_equations(self):
        self.links = {self.varlist.index(t) : list() for t in self.gt}
        for t in self.gt:
            for s in self.gt[t]:
                coeff = random.uniform(-0.99, 0.99)
                func = random.choice(FUNCTIONS)
                self.links[self.varlist.index(t)].append(((self.varlist.index(s[0]), s[1]), coeff, func))

        return self.links
       

    def look_for_source(self, gt, source, target):
        confounded = list()
        for t in gt.keys():
            for s in gt[t]:
                if s[0] != target and s[0] == source[0] and s[1] != source[1]:
                    confounded.append({s[0]: [(target, source[0], source[1]), (t, s[0], s[1])]})
        return confounded


    def get_lagged_confounders(self, gt):
        confounders = list()
        for t in gt.keys():
            for s in gt[t]:
                if s[0] != t:
                    tmp  = deepcopy(gt)
                    del tmp[t]
                    confounded = self.look_for_source(tmp, s, t)
                    if not isinstance(confounded, list): confounded = list(confounded)
                    if confounded:
                        for c in confounded:
                            if not self.exists(confounders, c):
                                confounders.append(c)
        return confounders


    def exists(self, confs, c):
        for conf in confs:
            c1 = list(conf.keys())[0]
            c2 = list(c.keys())[0]
            if c1 == c2 and conf[c1][0] == c[c2][1] and conf[c1][1] == c[c2][0]: 
                return True
        return False
    
    def gen_obs_ts(self):
        if self.links is None: self.get_equations()
        data, _ = toys.structural_causal_process(self.links, T = self.T)
        
        return data
    
    def gen_int_ts(self, int_var, int_val, int_len):
        if self.links is None: self.get_equations()

        intervention = int_val*np.ones(int_len)
        intervention_data, _ = toys.structural_causal_process(self.links, T=int_len, noises=None, seed=7,
                                                              intervention = {self.varlist.index(int_var) : intervention}, 
                                                              intervention_type='hard')
        return intervention_data
