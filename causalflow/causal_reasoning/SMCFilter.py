import copy
import itertools
import numpy as np
from scipy.stats import multivariate_normal
from causalflow.basics.constants import DataType, NodeType
import causalflow.causal_reasoning.DensityUtils as DensityUtils
import random

from causalflow.graph import DAG

class SMCFilter:
    def __init__(self, dbn, num_particles=500, max_components=50):
        """
        Sequential Monte Carlo (Particle Filtering) for inference in a Dynamic Bayesian Network.

        Args:
            dbn (DynamicBayesianNetwork): The Bayesian network object with precomputed densities.
            num_particles (int): Number of particles for the filter.
        """
        np.random.seed(42)
        random.seed(42)

        self.dbn = dbn.dbn
        self.dag : DAG = dbn.dag
        self.data_type = dbn.data_type
        self.node_type = dbn.node_type
        self.max_lag = dbn.dag.max_lag
        self.max_components = max_components
        
        self.num_particles = num_particles
        self.particles = {lag: {node: {context: None for context in self.dbn[node].keys()} for node in self.dbn.keys()} for lag in range(-abs(self.max_lag), 1)}
        self.distributions = {lag: {node: {context: {'p':None, 'parents':None} for context in self.dbn[node].keys()} for node in self.dbn.keys()} for lag in range(-abs(self.max_lag), 1)}
        self.init_weights()

        
        
    def sample_particles(self, t, intervention_set):
        topo_order = [node for node in self.dag.get_topological_order() if -abs(node[1]) == t]

        for node, lag in topo_order:
            for context in self.dbn[node].keys():
                if self.particles[t][node][context] is not None: continue
                # #! **If the variable is intervened upon (real intervention), assign its value directly**
                # if intervention_set is not None and (node, lag) in intervention_set:
                #     self.particles[t][node][context] = np.full(self.num_particles, intervention_set[(node, lag)])
                #     continue  # Skip sampling
                parents = self.get_parents((node, lag), check_lags=False)
                system_parents = [p for p in parents if self.node_type[p[0]] is not NodeType.Context]
                if not parents or all([-abs(p[1]) < -abs(self.max_lag) for p in system_parents]): 
                    self.particles[t][node][context] = self.gen_particles((node, lag), context, self.dbn[node][context].pY, self.num_particles)
                    self.distributions[t][node][context]['p'] = [self.dbn[node][context].pY]*self.num_particles
                else:      
                    parent_context = {}
                    for p in system_parents:
                        if p[0] == node: 
                            parent_context[p[0]] = context
                            continue
                        for pc in self.dbn[p[0]].keys():
                            if pc != ():
                                intersection = set(context).intersection(set(pc))
                                if intersection:
                                    parent_context[p[0]] = DensityUtils.format_combo([c for c in intersection])
                            else:
                                parent_context[p[0]] = pc
                    parent_values = {p: None for p in system_parents}
                    for p in system_parents:
                        pc = parent_context[p[0]]
                        if -abs(p[1]) >= -abs(self.max_lag) and self.particles[-abs(p[1])][p[0]][pc] is not None:
                            parent_values[p] = self.particles[-abs(p[1])][p[0]][pc]
                        else:
                            parent_values[p] = self.gen_particles(p, pc, self.dbn[p[0]][context].pY, self.num_particles)
                    new_particles = np.zeros(self.num_particles)
                    new_distrs = []
                    new_parents = []
                    for i in range(self.num_particles):
                        tmp_parent_values = {p[0]: parent_values[p][i] for p in system_parents}
                        pY_gX = self.dbn[node][context].get_pY_gX(given_p=tmp_parent_values if parent_values else None)
                        new_distrs.append(pY_gX)
                        new_parents.append(tmp_parent_values)
                        new_particles[i] = self.gen_particles((node, lag), pc, pY_gX, 1)
                        
                    self.particles[-abs(lag)][node][context] = new_particles
                    self.distributions[-abs(lag)][node][context]['p'] = new_distrs
                    self.distributions[-abs(lag)][node][context]['parents'] = new_parents

        
    def init_weights(self):
        self.weights = np.ones(self.num_particles) / self.num_particles  # Initial uniform weights


    def sample_from_gmm(self, gmm_params, num_samples):
        """Sample particles from a given GMM distribution."""
        means, covariances, weights = np.array(gmm_params["means"], dtype=np.float64), \
                                      np.array(gmm_params["covariances"], dtype=np.float64), \
                                      np.array(gmm_params["weights"], dtype=np.float64)
        # means, covariances, weights = gmm_params["means"], gmm_params["covariances"], gmm_params["weights"]
        chosen_components = np.random.choice(len(weights), size=num_samples, p=weights)
        samples = np.array([np.random.multivariate_normal(means[i], covariances[i]) for i in chosen_components])
        return samples.squeeze()
    
    
    def sample_from_categorical(self, values ,categorical_params, num_samples):
        """Sample particles for discrete variables from a categorical distribution."""
        weights = categorical_params["weights"]  # Probabilities for each value
        samples = np.random.choice(values, size=num_samples, p=weights)
        return samples
    
    
    def update_weights(self, target_node, given_values, given_context, adjustment_set=None):
        """
        Generalized weight update function for Sequential Monte Carlo (SMC).

        - Step 1: Compute likelihood P(X_t | Parent(X_t)) for all intermediate nodes.
        - Step 2: Apply correction using observed evidence at previous time step (if available).
        - Step 3: Normalize the weights.

        Args:
            given_values (dict): Dictionary of observed variables and their values.
            given_context (dict): Context information for evidence variables.
        """
        if adjustment_set is None:
            self._observational_update_weights(target_node, given_values, given_context)
            return
        
        ### **Step 1: Compute likelihood P(X_t | Parent(X_t)) for each Z**
        for adj in adjustment_set:
            adj_context = self.get_context(adj, given_context)
            ADJ_particles = self.particles[-abs(adj[1])][adj[0]][adj_context]
            unique_Z_values, P_Z = np.unique(ADJ_particles, return_counts=True)
            P_Z = P_Z / np.sum(P_Z)  # Normalize to get probability
        
        # Initialize likelihoods
        adjustment_likelihoods = np.zeros((len(unique_Z_values), self.num_particles))

        ADJ_particles = self.particles[-abs(adj[1])][adj[0]][adj_context]
        for j, z_val in enumerate(unique_Z_values):
            joint_index = {}
            # for evidence in given_values:
            #     evidence_context = self.get_context(evidence, given_context)
            #     particles = self.particles[evidence[1]][evidence[0]][evidence_context]
            #     joint_index.update({evidence: np.where(np.abs(particles - given_values[evidence] < 1e-3))[0].tolist()})
            joint_index.update({adjustment_set[0]: np.where(ADJ_particles == z_val)[0].tolist()})
            joint_index_set = list(set.intersection(*map(set, joint_index.values())))
            joint_likelihoods_Z = np.ones(self.num_particles)  # Separate likelihood for each Z
            
            all_relevant_nodes = set(self.get_intermediate_nodes(target_node, given_values.keys()))
            for evidence in given_values.keys():
                for a in self.dag.get_anchestors(evidence[0], include_lag=True):
                    all_relevant_nodes.add((a[0], -abs(a[1])))
            all_relevant_nodes = [node for node in all_relevant_nodes if self.node_type[node[0]] != NodeType.Context]
            all_relevant_nodes = sorted(all_relevant_nodes, key=lambda n: self.dag.get_topological_order().index(n))

            for node in all_relevant_nodes:
                node_name, node_lag = node
                    
                node_context = self.get_context(node, given_context)
                X_particles = self.particles[node_lag][node_name][node_context]
                X_p = self.distributions[node_lag][node_name][node_context]['p']

                likelihoods_X_t = np.zeros(self.num_particles)
                for i in joint_index_set:
                    # Compute likelihood for given Z
                    likelihoods_X_t[i] = np.sum([
                        X_p[i]["weights"][k] * multivariate_normal.pdf(X_particles[i], mean=X_p[i]["means"][k], cov=X_p[i]["covariances"][k]) 
                        for k in range(len(X_p[i]["weights"]))
                    ])

                likelihoods_X_t += 1e-300
                joint_likelihoods_Z *= likelihoods_X_t  # Multiply into likelihood chain

            # Multiply by P(Z)
            adjustment_likelihoods[j, :] += joint_likelihoods_Z * P_Z[j]
            
        adjustment_likelihoods = np.sum(adjustment_likelihoods, axis=0)
        
        ### **Step 2: Compute Observation Likelihood P(O_t | X_t)**
        observation_likelihoods = np.ones(self.num_particles)

        for evidence_var, observed_value in given_values.items():
            evidence_context = self.get_context(evidence_var, given_context)

            # Retrieve the particle values for this evidence variable
            particle_values = self.particles[evidence_var[1]][evidence_var[0]][evidence_context]

            # Compute likelihood based on how close the observed evidence is to the sampled particles
            likelihoods_O_t = np.zeros(self.num_particles)
            for i in range(self.num_particles):
                likelihoods_O_t[i] = multivariate_normal.pdf(
                    observed_value, mean=particle_values[i], cov=np.var(particle_values) + 1e-6
                )
            observation_likelihoods *= likelihoods_O_t

        ### **Step 3: Normalize Weights**
        self.weights *= adjustment_likelihoods * observation_likelihoods
        # Normalize weights safely
        total_weight = np.sum(self.weights)
        if total_weight == 0 or np.isnan(total_weight):  
            self.weights = np.ones_like(self.weights) / len(self.weights)  # Reset to uniform weights
        else:
            self.weights /= total_weight  # Normalize properly
        
    def _observational_update_weights(self, target_node, given_values, given_context):
        """
        Generalized weight update function for Sequential Monte Carlo (SMC).

        - Step 1: Compute likelihood P(X_t | Parent(X_t)) for all intermediate nodes.
        - Step 2: Apply correction using observed evidence at previous time step (if available).
        - Step 3: Normalize the weights.

        Args:
            given_values (dict): Dictionary of observed variables and their values.
            given_context (dict): Context information for evidence variables.
        """
        # joint_index = {}
        # for evidence in given_values:
        #     evidence_context = self.get_context(evidence, given_context)
        #     particles = self.particles[evidence[1]][evidence[0]][evidence_context]
        #     joint_index.update({evidence: np.where(np.abs(particles - given_values[evidence] < 1e-3))[0].tolist()})
        # joint_index_set = list(set.intersection(*map(set, joint_index.values())))
            
        # Initialize joint likelihoods for all particles
        joint_likelihoods = np.ones(self.num_particles)

        ### **Step 1: Compute Likelihood P(X_t | Parent(X_t)) for All Intermediate Nodes**
        all_relevant_nodes = set(self.get_intermediate_nodes(target_node, given_values.keys()))
        for evidence in given_values.keys():
            for a in self.dag.get_anchestors(evidence[0], include_lag=True):
                all_relevant_nodes.add((a[0], -abs(a[1])))
        all_relevant_nodes.add((target_node, 0))
        all_relevant_nodes = [node for node in all_relevant_nodes if self.node_type[node[0]] != NodeType.Context]
        all_relevant_nodes = sorted(all_relevant_nodes, key=lambda n: self.dag.get_topological_order().index(n))
        for node in all_relevant_nodes:
            node_name, node_lag = node
            node_context = self.get_context(node, given_context)

            # Extract sampled values for this node
            X_particles = self.particles[node_lag][node_name][node_context]
            X_p = self.distributions[node_lag][node_name][node_context]['p']

            # Compute likelihood P(X_t | Parent(X_t)) for each particle
            likelihoods_X_t = np.zeros(self.num_particles)
            # for i in joint_index_set:
            for i in range(self.num_particles):
                likelihoods_X_t[i] = np.sum([
                    X_p[i]["weights"][k] * multivariate_normal.pdf(X_particles[i], mean=X_p[i]["means"][k], cov=X_p[i]["covariances"][k]) 
                    for k in range(len(X_p[i]["weights"]))
                ])

            # Avoid numerical issues
            likelihoods_X_t += 1e-300

            # Multiply likelihoods into joint likelihoods
            joint_likelihoods *= likelihoods_X_t

        ### **Step 2: Compute Observation Likelihood P(O_t | X_t)**
        final_likelihoods = np.ones(self.num_particles)

        for evidence_var, observed_value in given_values.items():
            evidence_context = self.get_context(evidence_var, given_context)

            # Retrieve the particle values for this evidence variable
            particle_values = self.particles[evidence_var[1]][evidence_var[0]][evidence_context]

            # Compute likelihood based on how close the observed evidence is to the sampled particles
            likelihoods_O_t = np.zeros(self.num_particles)
            for i in range(self.num_particles):
                likelihoods_O_t[i] = multivariate_normal.pdf(
                    observed_value, mean=particle_values[i], cov=np.var(particle_values) + 1e-6
                )
            final_likelihoods *= likelihoods_O_t

        # Apply final likelihood correction
        joint_likelihoods *= final_likelihoods

        ### **Step 3: Normalize Weights**
        self.weights *= joint_likelihoods
        # Normalize weights safely
        total_weight = np.sum(self.weights)
        if total_weight == 0 or np.isnan(total_weight):  
            self.weights = np.ones_like(self.weights) / len(self.weights)  # Reset to uniform weights
        else:
            self.weights /= total_weight  # Normalize properly
          
    
    # def resample(self):
    #     """
    #     Resample particles at the most recent time step (t=0) using systematic resampling.
    #     Ensures particle diversity while avoiding degeneration.
    #     """
    #     indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)

    #     # Only resample at t=0
    #     t = 0  
    #     for node, contexts in self.particles[t].items():
    #         for context, particles in contexts.items():
    #             if particles is not None:
    #                 self.particles[t][node][context] = particles[indices]
    #                 self.distributions[t][node][context]['p'] = [self.distributions[t][node][context]['p'][i] for i in indices]
    #                 if self.distributions[t][node][context]['parents'] is not None:
    #                     self.distributions[t][node][context]['parents'] = [self.distributions[t][node][context]['parents'][i] for i in indices]

    #     # Reset weights after resampling
    #     self.init_weights()
    def resample(self):
        """
        Perform Adaptive Stratified Resampling.
        - Computes the Effective Sample Size (ESS).
        - Resamples only when necessary.
        - Uses Stratified Resampling to improve diversity.
        """
        # Compute Effective Sample Size (ESS)
        ESS = 1.0 / np.sum(self.weights ** 2)  
        threshold = self.num_particles / 2  # Resample if ESS falls below half

        if ESS < threshold:  # Only resample if necessary
            # Compute cumulative sum of weights
            cumulative_sum = np.cumsum(self.weights)
            cumulative_sum[-1] = 1.0  # Fix rounding errors

            # Stratified positions (split weight space evenly)
            positions = (np.arange(self.num_particles) + np.random.uniform(0, 1, self.num_particles)) / self.num_particles
            indices = np.searchsorted(cumulative_sum, positions)

            t = 0  # Only resample for the most recent timestep
            for node, contexts in self.particles[t].items():
                for context, particles in contexts.items():
                    if particles is not None:
                        self.particles[t][node][context] = particles[indices]
                        self.distributions[t][node][context]['p'] = [self.distributions[t][node][context]['p'][i] for i in indices]
                        if self.distributions[t][node][context]['parents'] is not None:
                            self.distributions[t][node][context]['parents'] = [self.distributions[t][node][context]['parents'][i] for i in indices]

            # Reset weights to be uniform after resampling
            self.init_weights()


    def estimate_posterior(self, node, given_values):
        """Estimate the posterior distribution P(X_t | evidence) by fitting a GMM to the particles."""
        given_context = {n[0]: given_values[n] for n in given_values.keys() if self.node_type[n[0]] == NodeType.Context}
        node_context = self.get_context((node, 0), given_context)

        gmm = DensityUtils.fit_gmm(data=self.particles[0][node][node_context].reshape(-1, 1),
                                   caller="p(Y|X)",
                                   max_components=self.max_components, 
                                   standardize=True)
        return gmm


    def compute_expectation(self, node, node_context):
        """Compute the expected value E[X_t | evidence]."""
        return np.sum(self.particles[0][node][node_context] * self.weights) / np.sum(self.weights)

    
    def get_parents(self, node, check_lags=True):
        parents = []
        node_name, node_lag = node
        for s in self.dag.g[node_name].sources.keys():
            s_name, s_lag = s
            if check_lags and (abs(s_lag) + abs(node_lag) > self.dag.max_lag): continue
            parents.append((s_name, -(abs(s_lag) + abs(node_lag))))
        return parents
    
    
    def get_intermediate_nodes(self, target_node, evidence_nodes):
        """Find all intermediate nodes that lie on the path from evidence to target."""
        intermediate_nodes = set()
        parents = self.get_parents((target_node, 0))
        queues = {i: [parent] for i, parent in enumerate(parents)}
        new_queues = copy.copy(queues)
        while queues:
            for k, path in queues.items():
                parent = path[-1]
                if parent not in evidence_nodes:  # Stop if we reach evidence
                    for pp in self.get_parents(parent):
                        new_queues[len(new_queues)] = path + [pp]
                    new_queues.pop(k)
                else:
                    for parent in path: intermediate_nodes.add(parent)
                    new_queues.pop(k)
                    continue
            queues = copy.copy(new_queues)
        return list(intermediate_nodes)
    
    
    def get_evidence_target_chain(self, target_node, evidence_nodes):
        """Find all intermediate nodes that lie on the path from evidence to target."""
        intermediate_nodes = {}
        parents = self.get_parents((target_node, 0))
        queues = {i: [parent] for i, parent in enumerate(parents)}
        new_queues = copy.copy(queues)
        while queues:
            for k, path in queues.items():
                parent = path[-1]
                if parent not in evidence_nodes:  # Stop if we reach evidence
                    for pp in self.get_parents(parent):
                        new_queues[len(new_queues)] = path + [pp]
                    new_queues.pop(k)
                else:
                    intermediate_nodes.update({len(intermediate_nodes): path})
                    new_queues.pop(k)
                    continue
            queues = copy.copy(new_queues)
        for intermediate_node in intermediate_nodes.values():
            intermediate_node.reverse()
            intermediate_node.append((target_node, 0))
        return intermediate_nodes

    
    def gen_particles(self, node, context, p, num_particles):
        if self.data_type[node[0]] == DataType.Discrete:
            return self.sample_from_categorical(self.dbn[node[0]][context].y.unique_values, p, num_particles)
        else:
            return self.sample_from_gmm(p, num_particles)
        
        
    def get_context(self, node, given_context):
        anchestors = self.dag.get_anchestors(node[0])
        context_anchestors = [a for a in anchestors if self.node_type[a] == NodeType.Context]
        context = DensityUtils.format_combo([(c, given_context[c]) for c in context_anchestors])
        return context
    
    
    def shift_particles(self):
        """
        Shift all stored particles backward in time to maintain time consistency.

        - Particles at t=0 become particles at t=-1.
        - Particles at t=-1 become particles at t=-2, and so on.
        - The oldest time step (t=-max_lag) is removed.
        - Creates an empty placeholder for t=0.
        """
        for t in range(-abs(self.max_lag) + 1, 1):
            for node in self.particles[t].keys():
                for context in self.particles[t][node].keys():
                    self.particles[t-1][node][context] = self.particles[t][node][context]
                    self.distributions[t-1][node][context] = self.distributions[t][node][context]

        # Create an empty placeholder for new time step t=0
        self.particles[0][node][context] = None  # Ready for the next query step
        self.distributions[0][node][context] = {'p': None, 'parents': None}  # Ready for the next query step
                    
                    
    def particle_propagation(self, target_node, given_values, given_context, intervention_set = None):
        intermediate_nodes = self.get_intermediate_nodes(target_node, given_values.keys())
        intermediate_nodes.append((target_node, 0)) # Step 4: Sample the Target Node

        intermediate_nodes = sorted(intermediate_nodes, key=lambda node: self.dag.get_topological_order().index(node))

        # Step 3: Propagate Particles Following the Bayesian Structure
        for node in intermediate_nodes:
            # here I need to check if particles for this specific node and time-lag have already been sampled
            node_context = self.get_context(node, given_context)
                    
            if self.particles[-abs(node[1])][node[0]][node_context] is None:
                #! THIS IS USEFUl IF THE INTERVENTION IS REALLY PERFORMED
                # # If variable is intervened upon (do-operation), assign fixed value instead of sampling
                # if intervention_set is not None and node in intervention_set:
                #     self.particles[-abs(node[1])][node[0]][node_context] = np.full(self.num_particles, intervention_set[node])
                #     continue  # Skip sampling
                            
                # Special case: If the node is a context variable, sample directly (no prediction needed)
                if self.node_type[node[0]] == NodeType.Context:
                    self.particles[-abs(node[1])][node[0]][node_context] = self.gen_particles(node, 
                                                                                              node_context, 
                                                                                              self.dbn[node[0]][node_context].pY, 
                                                                                              self.num_particles)
                    self.distributions[-abs(node[1])][node[0]][node_context]['p'] = [self.dbn[node[0]][node_context].pY]*self.num_particles
                else:
                    parents = self.get_parents(node, check_lags=False)
                    if not parents:
                        self.particles[-abs(node[1])][node[0]][node_context] = self.gen_particles(node, 
                                                                                                  node_context, 
                                                                                                  self.dbn[node[0]][node_context].pY, 
                                                                                                  self.num_particles)
                        self.distributions[-abs(node[1])][node[0]][node_context]['p'] = [self.dbn[node[0]][node_context].pY]*self.num_particles

                    else:
                        parent_context = {p[0]: self.get_context(p, given_context) for p in parents}
                        
                        # Recursive Parent Validation: Ensure all parents have particles
                        for p in parents:
                            if self.particles[-abs(p[1])][p[0]][parent_context[p[0]]] is None:
                                self.particle_propagation(p[0], given_values, given_context)  # Compute parent first                         
                        new_particles = np.zeros(self.num_particles)
                        new_distrs = []
                        new_parents = []
                        # Iterate over each particle to compute the new state
                        for i in range(self.num_particles):
                            parent_values = {p[0]: self.particles[-abs(p[1])][p[0]][parent_context[p[0]]][i] 
                                            for p in parents 
                                            if self.node_type[p[0]] != NodeType.Context}
                            pY_gX = self.dbn[node[0]][node_context].get_pY_gX(given_p=parent_values if parent_values else None)
                            new_distrs.append(pY_gX)
                            new_parents.append(parent_values)
                            new_particles[i] = self.gen_particles(node, node_context, pY_gX, 1)
                        self.particles[-abs(node[1])][node[0]][node_context] = new_particles
                        self.distributions[-abs(node[1])][node[0]][node_context]['p'] = new_distrs
                        self.distributions[-abs(node[1])][node[0]][node_context]['parents'] = new_parents


    def query(self, target_node, given_values, given_context, intervention_set = None, adjustment_set = None):
        """
        Perform one step of SMC for an ancestor-based query P(target_node | ancestors).

        Args:
            target_node (str): The node to infer (e.g., ELT_t).
            given_values (dict): Dictionary of observed ancestor variables.
        """        
        # Step 1: Particle Propagation
        self.particle_propagation(target_node, given_values, given_context, intervention_set)
        
        # Step 2: Compute Importance Weights
        self.update_weights(target_node, given_values, given_context, adjustment_set)
        
        # Step 3: Expectation
        target_node_context = self.get_context((target_node, 0), given_context)
        e = self.compute_expectation(target_node, target_node_context)

        # Step 4: Fill Missing Particles
        self.sample_particles(0, intervention_set)
        
        # Step 5: Resample Particles
        self.resample()
                
        return e
    
    
    def sequential_query(self, target_node, given_values, given_context, num_steps, 
                         intervention_var = None, adjustment_set = None):
        """
        Perform sequential inference for multiple time steps.
        
        Args:
            target_node (str): The node to infer (e.g., ELT_t).
            given_values (dict): Dictionary of observed ancestor variables at the first time step.
            given_context (dict): Context information for evidence variables.
            num_steps (int): Number of time steps to perform sequential inference.

        Returns:
            list: List of expected values of `target_node` over `num_steps`.
        """
        if intervention_var is not None:
            if intervention_var not in given_values and intervention_var not in given_context:
                raise ValueError("Intervention value missing!")
            
            if intervention_var in given_values: intervention_set = {intervention_var: given_values[intervention_var]}
            elif intervention_var in given_context: intervention_set = {intervention_var: given_values[given_context]}
            if adjustment_set is not None and not isinstance(adjustment_set, list): adjustment_set = [adjustment_set]
        else:
            intervention_set = None
            adjustment_set = None
        self.sample_particles(-abs(self.max_lag), intervention_set)
        
        expectations = []

        for t in range(num_steps):
            given_v = {v: given_values[v][t] for v in given_values}
            # Perform one-step inference
            expected_value = self.query(target_node, given_v, given_context, intervention_set, adjustment_set)
            # expected_value = self.query(target_node, given_values, given_context, intervention_set, adjustment_set)
            
            # Store expectation
            expectations.append(expected_value)
            
            # Shift particles: move t to t-1
            self.shift_particles()
            
            # Prepare new given_values for the next step
            if (target_node, -1) in given_values and t+1 <= num_steps-1:
                given_values = copy.deepcopy(given_values)
                given_values[(target_node, -1)][t+1] = expected_value  # Use expectation as new evidence
            
        return expectations
