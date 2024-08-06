import os
import random
from causalflow.random_system.RandomGraph import NoiseType, RandomGraph
from pathlib import Path


if __name__ == '__main__':
    # Simulation params
    resdir = "AIS_major/S5_randomgraph"
    nfeature = 5
    
    # RandomDAG params 
    nsample_obs = 1250
    nsample_int = 250
    # nsample_obs = 750
    # nsample_int = 150
    min_c = 0.1
    max_c = 0.5
    link_density = 3
    max_exp = 2
    # functions = ['']
    # operators = ['+', '-']
    functions = ['','sin', 'cos', 'exp', 'abs', 'pow']
    operators = ['+', '-', '*', '/']
    n_hidden_confounders = 2
    
            
    Path(os.getcwd() + "/results/" + resdir).mkdir(parents=True, exist_ok=True)
    resfolder = 'results/' + resdir + '/' + str(nfeature)                 
        
    min_lag = 0
    max_lag = 3
    os.makedirs(resfolder, exist_ok = True)
                        
    # Noise params 
    noise_param = random.uniform(0.5, 2)
    noise_uniform = (NoiseType.Uniform, -noise_param, noise_param)
    noise_gaussian = (NoiseType.Gaussian, 0, noise_param)
    noise_weibull = (NoiseType.Weibull, 2, 1)
    RS = RandomGraph(nvars = nfeature, nsamples = nsample_obs + nsample_int, 
                   link_density = link_density, coeff_range = (min_c, max_c), max_exp = max_exp, 
                   min_lag = min_lag, max_lag = max_lag, noise_config = random.choice([noise_uniform, noise_gaussian, noise_weibull]),
                   functions = functions, operators = operators, n_hidden_confounders = n_hidden_confounders)
    RS.gen_equations()
    RS.ts_dag(withHidden = True, save_name = resfolder + '/gt_complete', randomColors=True)
    EQUATIONS = RS.print_equations()
    # Define the file path
    file_path = resfolder + '/equations.txt'

    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Check if EQUATIONS is a list and join it into a single string
        if isinstance(EQUATIONS, list):
            # Join list elements with new lines (assuming each equation should be on a new line)
            content = '\n'.join(EQUATIONS)
        else:
            content = EQUATIONS
        
        # Write the content to the file
        file.write(content)
    # RS.ts_dag(withHidden = False, save_name = resfolder + '/gt')       