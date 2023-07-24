import copy
from enum import Enum
import math
import random
import numpy as np
from fpcmci.preprocessing.data import Data


class NoiseType(Enum):
    Uniform = 0
    Gaussian = 1
    
    
class PriorityOp(Enum):
    M = '*'
    D = '/'
    

class RandomSystem:
    def __init__(self, nvars, nsamples, max_terms, coeff_range: tuple, 
                 max_exp, min_lag, max_lag, noise_config: tuple = None, 
                 operators = ['+', '-', '*'], 
                 functions = ['','sin', 'cos', 'exp', 'abs', 'pow'],
                 interv_perc: float = 1,
                 interv_randrange: tuple = None):
        
        self.N = nvars
        self.T = nsamples
        self.max_terms = max_terms
        self.coeff_range = coeff_range
        self.exponents = list(range(0, max_exp))
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.interv_T = int(self.T * interv_perc)
        self.interv_randrange = interv_randrange
        
        self.variables = ['X_' + str(i) for i in range(nvars)]
        self.operators = operators
        self.functions = functions
        self.equations = {var: list() for var in self.variables}
        self.noise_config = noise_config
        self.noise = None
        if noise_config is not None:
            if noise_config[0] is NoiseType.Uniform:
                self.noise = np.random.uniform(noise_config[1], noise_config[2], (self.T, self.N))
            elif noise_config[1] is NoiseType.Gaussian:
                self.noise = np.random.normal(noise_config[1], noise_config[2], (self.T, self.N))


    def gen_equations(self):
        """
        Generates random equations using the operator and function lists provided in the constructor 
        """
        for var in self.variables:
            equation = []
            var_choice = copy.deepcopy(self.variables)
            for _ in range(random.randint(1, self.max_terms)):
                coefficient = random.uniform(self.coeff_range[0], self.coeff_range[1])
                variable = random.choice(var_choice)
                var_choice.remove(variable)
                operator = random.choice(self.operators)
                lag = random.randint(self.min_lag, self.max_lag)
                function = random.choice(self.functions)
                if function == 'pow':
                    exponent = random.choice(self.exponents)
                    term = (operator, coefficient, function, variable, exponent, lag)
                else:
                    term = (operator, coefficient, function, variable, lag)
                equation.append(term)
            self.equations[var] = equation


    def print_equations(self):
        """
        Prints the generated equations
        """
        for target, eq in self.equations.items():
            equation_str = target + '(t) = '
            for i, term in enumerate(eq):
                if len(term) == 6:
                    operator, coefficient, function, variable, exponent, lag = term
                    if i != 0: 
                        term_str = f"{operator} {coefficient} * {function}({variable}, {exponent})(t-{lag}) "
                    else:
                        term_str = f"{coefficient} * {function}({variable}, {exponent})(t-{lag}) "
                else:
                    operator, coefficient, function, variable, lag = term
                    if function != '':
                        if i != 0: 
                            term_str = f"{operator} {coefficient} * {function}({variable})(t-{lag}) "
                        else:
                            term_str = f"{coefficient} * {function}({variable})(t-{lag}) "
                    else:
                        if i != 0: 
                            term_str = f"{operator} {coefficient} * {variable}(t-{lag}) "
                        else:
                            term_str = f"{coefficient} * {variable}(t-{lag}) "
                        
                equation_str += term_str
            print(equation_str)
            
            
    def __evaluate_term(self, term, t, data):
        """
        Evaluates single term componing an equation

        Args:
            term (tuple): term to evaluate
            t (int): time step
            data (numpy array): time-series

        Returns:
            tuple: operator and value of the term
        """
        operator, coefficient, function, variable, *args = term
        if function == '':
            lag = args[0]
            term_value = coefficient * (data[t - lag, self.variables.index(variable)])
        elif function == 'pow':
            exponent, lag = args
            term_value = coefficient * data[t - lag, self.variables.index(variable)] ** exponent
        elif function == 'abs':
            lag = args[0]
            term_value = coefficient * abs(data[t - lag, self.variables.index(variable)])
        else:
            lag = args[0]
            term_value = coefficient * getattr(math, function)(data[t - lag, self.variables.index(variable)])
        return operator, term_value
    
    
    def __handle_priority_operator(self, eq):
        """
        Evaluates all the terms with operato * ans /

        Args:
            eq (list): equation (list of term)

        Returns:
            list: equation with all * and / evaluated
        """
        op = '*'
        while (op in eq):
            op_i = eq.index(op)
            op1_i = op_i - 1
            op2_i = op_i + 1
            eq[op1_i] = eq[op1_i] * eq[op2_i]
                
            indices_set = set([op_i, op2_i])
            eq = [item for i, item in enumerate(eq) if i not in indices_set]
            
        op = '/'
        while (op in eq):
            op_i = eq.index(op)
            op1_i = op_i - 1
            op2_i = op_i + 1
            eq[op1_i] = eq[op1_i] / eq[op2_i]
                
            indices_set = set([op_i, op2_i])
            eq = [item for i, item in enumerate(eq) if i not in indices_set]
            
        return eq
     

    def __evaluate_equation(self, equation, t, data = None):
        """
        Evaluates equation

        Args:
            equation (list): equation (list of term)
            t (int): time step

        Returns:
            float: equation value
        """
        eq = list()
        for i, term in enumerate(equation):
            if data is None: data = self.np_data
            operator, term = self.__evaluate_term(term, t, data)
            if i == 0:
                eq.append(term)
            else:
                eq.append(operator)
                eq.append(term)
                
        # Handle * and / before + and -
        eq = self.__handle_priority_operator(eq)
        
        equation_value = eq.pop(0)
        for i in range(0, len(eq), 2):
            op = eq[i]
            term = eq[i+1]
            if op == '+': equation_value = equation_value + term
            elif op == '-': equation_value = equation_value - term
        return equation_value


    def gen_obs_ts(self):
        """
        Generates time-series data

        Returns:
            Data: generated data
        """
        self.np_data = np.zeros((self.T, self.N))
        for t in range(self.max_lag, self.T):
            for target, eq in self.equations.items():
                self.np_data[t, self.variables.index(target)] = self.__evaluate_equation(eq, t)
                if self.noise is not None: self.np_data[t, self.variables.index(target)] += self.noise[t, self.variables.index(target)]
                    
        self.data = Data(self.np_data, self.variables)
        return self.data
    
    
    def gen_interv_ts(self, interventions):
        """
        Generates time-series corresponding to intervention(s)

        Args:
            interventions (dict): dictionary {INT_VAR : {INT_LEN: int_len, INT_VAL: int_val}}

        Returns:
            Data: interventional time-series data
        """
                
        self.int_data = dict()
        for int_var in interventions:
            T = int(interventions[int_var]["T"])
            if self.noise_config is not None:
                if self.noise_config[0] is NoiseType.Uniform:
                    int_noise = np.random.uniform(self.noise_config[1], self.noise_config[2], (T, self.N))
                elif self.noise_config[1] is NoiseType.Gaussian:
                    int_noise = np.random.normal(self.noise_config[1], self.noise_config[2], (T, self.N))
            np_int_data = np.zeros((T, self.N))
            for t in range(self.max_lag, T):
                for target, eq in self.equations.items():
                    if target != int_var:
                        np_int_data[t, self.variables.index(target)] = self.__evaluate_equation(eq, t, np_int_data)
                        if self.noise_config is not None: np_int_data[t, self.variables.index(target)] += int_noise[t, self.variables.index(target)]
                    else:
                        np_int_data[t, self.variables.index(target)] = interventions[int_var]["VAL"]
                        
            self.int_data[int_var] = Data(np_int_data, self.variables)
        return self.int_data
    
    
    def get_SCM(self):
        """
        Outputs the Structural Causal Model

        Returns:
            dict: scm
        """
        scm = {target : list() for target in self.equations.keys()}
        for target, eq in self.equations.items():
            for term in eq:
                if len(term) == 6:
                    _, _, _, variable, _, lag = term
                else:
                    _, _, _, variable, lag = term
                if variable not in scm.keys(): continue # NOTE: this is needed when you hide a var
                scm[target].append((variable, -abs(lag)))
        return scm
    
    
    def print_SCM(self):
        """
        Prints the Structural Causal Model
        """
        scm = self.get_SCM()
        for t in scm: print(t + ' : ' + str(scm[t]))    
    
    
    def hide_var(self, var):
        """
        Hides a variable

        Args:
            var (str): variable name

        Raises:
            ValueError: if the specified variable is the one used for the intervention
        """
        if var in self.int_data:
            raise ValueError("Cannot hide the specified variable since it is used as intervention variable.")
        self.variables.remove(var)
        del self.equations[var]
        self.data.shrink(self.variables)
        
        for var in self.int_data:
            self.int_data[var].shrink(self.variables)
        
        
    def interv_var(self, int_var, int_len, int_value):
        """
        Generates intervention on a single variable

        Args:
            int_var (str): variable name
            int_len (int): intervention length
            int_value (float): intervention value

        Returns:
            Data: interventional time-series data
        """
        return self.gen_interv_ts({int_var: {"T": int_len, "VAL": int_value}})


    def get_lagged_confounders(self):
        """
        Returns lagged confounders list

        Returns:
            list: list of lagged confounders
        """
        gt = self.get_SCM()
        confounders = list()
        for t in gt.keys():
            for s in gt[t]:
                if s[0] != t:
                    tmp  = copy.deepcopy(gt)
                    del tmp[t]
                    confTriples = self.__lookForSource(tmp, s, t)
                    if not isinstance(confTriples, list): confTriples = list(confTriples)
                    if confTriples:
                        for confTriple in confTriples:
                            if not self.__exists(confounders, confTriple) and not self.__isThereALink(gt, confTriple):
                                confounders.append(confTriple)
        return confounders

        
    def __isThereALink(self, gt, confTriple):
        """
        Returns True if there is a link between the two confounded variables. False otherwise

        Args:
            gt (dict): structural causal model
            confTriple (str): confounder triple

        Returns:
            bool: Returns True if there is a link between the two confounded variables. False otherwise
        """

        confounder = next(iter(confTriple.keys()))
        int_var = min(confTriple[confounder], key=lambda x: abs(x[2]))[0]
        other = max(confTriple[confounder], key=lambda x: abs(x[2]))[0]
        for s in gt[other]:
            if s[0] == int_var: return True
        return False
    
    
    def __lookForSource(self, gt, source, target):
        """
        Searches the second confounded variable given the first and the confounder

        Args:
            gt (dict): structural causal model
            source (str): confounded variable
            target (str): confounder

        Returns:
            list: list of confonded variables
        """
        confounded = list()
        for t in gt.keys():
            for s in gt[t]:
                if s[0] != t and s[0] != target and s[0] == source[0] and s[1] != source[1]:
                    confounded.append({s[0]: [(target, source[0], source[1]), (t, s[0], s[1])]})
        return confounded


    def __exists(self, confs, c):
        """
        Checks if the new confounder has been already considered

        Args:
            confs (list): confounders
            c (dict): new confounder

        Returns:
            bool: True if the confounder is already in the list. False otherwise
        """
        for conf in confs:
            c1 = list(conf.keys())[0]
            c2 = list(c.keys())[0]
            if c1 == c2 and conf[c1][0] == c[c2][1] and conf[c1][1] == c[c2][0]: 
                return True
        return False


# # Example usage
# N = 5
# T = 1500
# noise = (NoiseType.Uniform, -0.1, 0.1)

# RS = RandomSystem(nvars = N, nsamples = T, 
#                   max_terms = 3, coeff_range = (-0.5, 0.5), max_exp = 2, 
#                   min_lag = 1, max_lag = 2, noise = noise,
#                   functions = [''])

# RS.gen_equations()
# RS.print_SCM()

# # Observational data
# obs_data = RS.gen_obs_ts()

# # Confounders list
# confounders = RS.get_lagged_confounders()
# print(confounders)
# sel_confounder = random.choice(confounders)
# print(sel_confounder)
# conf_to_hide = next(iter(sel_confounder.keys()))
# int_var = min(sel_confounder[conf_to_hide], key=lambda x: abs(x[2]))[0]

# # Interventional data
# int_data = RS.interv_var(int_var, T, 5)

# # Hide variable
# RS.hide_var(conf_to_hide)

# obs_data.plot_timeseries()
# int_data[int_var].plot_timeseries()
