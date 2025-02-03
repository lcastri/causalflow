import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from causalflow.basics.constants import DataType, NodeType
import causalflow.causal_reasoning.DensityUtils as DensityUtils

class SMCFilter:
    def __init__(self, dbn, num_particles=500):
        """
        Sequential Monte Carlo (Particle Filtering) for inference in a Dynamic Bayesian Network.

        Args:
            dbn (DynamicBayesianNetwork): The Bayesian network object with precomputed densities.
            num_particles (int): Number of particles for the filter.
        """
        self.dbn = dbn.dbn
        self.dag = dbn.dag
        self.data_type = dbn.data_type
        self.node_type = dbn.node_type
        self.num_particles = num_particles
        self.particles = {node: {} for node in self.dbn.keys()}
        for node in self.dbn.keys():
            for context in self.dbn[node].keys():
                if self.data_type[node] == DataType.Continuous:
                    self.particles[node][context] = self.sample_from_gmm(self.dbn[node][context].pY, 
                                                                         self.num_particles)
                else:
                    self.particles[node][context] = self.sample_from_categorical(self.dbn[node][context].y.unique_values, 
                                                                                 self.dbn[node][context].pY, 
                                                                                 self.num_particles)
        self.weights = np.ones(num_particles) / num_particles  # Initial uniform weights

    def sample_from_gmm(self, gmm_params, num_samples):
        """Sample particles from a given GMM distribution."""
        means, covariances, weights = gmm_params["means"], gmm_params["covariances"], gmm_params["weights"]
        chosen_components = np.random.choice(len(weights), size=num_samples, p=weights)
        samples = np.array([np.random.multivariate_normal(means[i], covariances[i]) for i in chosen_components])
        return samples
    
    def sample_from_categorical(self, values ,categorical_params, num_samples):
        """Sample particles for discrete variables from a categorical distribution."""
        weights = categorical_params["weights"]  # Probabilities for each value
        samples = np.random.choice(values, size=num_samples, p=weights)
        return samples

    def predict(self, node):
        """Propagate particles using the transition density P(X_t | X_t-1)."""
        density_model = self.dbn[node]
        self.particles[node] = self.sample_from_gmm(density_model.pJoint, self.num_particles)

    def update(self, node, observation):
        """Update particle weights using the observation likelihood P(O_t | X_t)."""
        density_model = self.dbn[node]
        likelihoods = np.array([
            multivariate_normal.pdf(observation, mean=density_model.pY["means"][i], cov=density_model.pY["covariances"][i])
            for i in range(len(density_model.pY["weights"]))
        ])
        self.weights *= likelihoods
        self.weights += 1e-300  # Avoid zero weights
        self.weights /= np.sum(self.weights)  # Normalize

    def resample(self):
        """Resample particles based on their weights using systematic resampling."""
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
        for node in self.particles:
            self.particles[node] = self.particles[node][indices]
        self.weights = np.ones(self.num_particles) / self.num_particles  # Reset weights

    def estimate_posterior(self, node):
        """Estimate the posterior distribution P(X_t | evidence) by fitting a GMM to the particles."""
        gmm = GaussianMixture(n_components=3, covariance_type="full")
        gmm.fit(self.particles[node])
        return gmm

    def compute_expectation(self, node):
        """Compute the expected value E[X_t | evidence]."""
        return np.average(self.particles[node], axis=0, weights=self.weights)

    def compute_map_estimate(self, node):
        """Compute the most probable state (MAP estimate)."""
        return self.particles[node][np.argmax(self.weights)]
    
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
        queue = [(target_node, 0)]  # Start from the target node

        while queue:
            current_node = queue.pop(0)
            parents = self.get_parents(current_node)

            for parent in parents:
                if parent not in evidence_nodes:  # Stop if we reach evidence
                    queue.append(parent)
                    intermediate_nodes.add(parent)

        return list(intermediate_nodes)

    def step_ancestor_query(self, target_node, given_values):
        """
        Perform one step of SMC for an ancestor-based query P(target_node | ancestors).

        Args:
            target_node (str): The node to infer (e.g., ELT_t).
            given_values (dict): Dictionary of observed ancestor variables.
        """
        # Context definition
        given_context = {node[0]: given_values[node] for node in given_values.keys() if self.node_type[node[0]] == NodeType.Context}
        
        # Identify intermediate nodes needed to connect evidence to the target
        intermediate_nodes = self.get_intermediate_nodes(target_node, given_values.keys())

        # Sample full joint distribution P(X) by propagating particles
        for node in intermediate_nodes:
            parents = self.get_parents(node, check_lags=False)
            node_context = DensityUtils.format_combo([(p[0], given_context[p[0]]) for p in parents if self.node_type[p[0]] == NodeType.Context])
            parent_context = {}
            if parents:
                for p in parents:
                    if self.get_parents(p, check_lags=False):
                        for pp in self.get_parents(p, check_lags=False):
                            if self.node_type[pp[0]] == NodeType.Context:
                                parent_context[p] = DensityUtils.format_combo((pp[0], given_values[pp]))
                    else:
                        parent_context[p[0]] = ()
                        
            # Special case: If the node is a context variable, sample directly (no prediction needed)
            if self.node_type[node[0]] == NodeType.Context:
                if node[0] in given_values:
                    # Use evidence values directly if given
                    self.particles[node[0]][node_context] = np.full(self.num_particles, given_values[node[0]])
                else:
                    # Otherwise, sample from prior pY
                    if self.data_type[node[0]] == DataType.Discrete:
                        self.particles[node[0]][node_context] = self.sample_from_categorical(self.dbn[node[0]][node_context].y.unique_values, 
                                                                                            self.dbn[node[0]][node_context].pY, 
                                                                                            self.num_particles)
                    else:
                        self.particles[node[0]][node_context] = self.sample_from_gmm(self.dbn[node[0]][node_context].pY, self.num_particles)
            else:
                # Iterate over each particle to compute the new state
                new_particles = np.zeros(self.num_particles)
                for i in range(self.num_particles):
                    parent_values = {p: self.particles[p[0]][i] for p in parents if self.node_type[p[0]] != NodeType.Context}
                    new_particles[i] = self.dbn[node[0]][node_context].predict(given_p=parent_values if parent_values else None)

                self.particles[node[0]][node_context] = new_particles

            self.particles[node[0]][node_context] = new_particles

        # Sample the target node using its parents
        parent_values = {p: self.particles[p] for p in self.dbn[target_node].parents.keys()}
        self.particles[target_node] = self.dbn[target_node].predict(given_p=parent_values)

        # Compute Importance Weights
        for evidence_var, observed_value in given_values.items():
            likelihoods = np.array([
                multivariate_normal.pdf(observed_value, mean=self.dbn[evidence_var].pY["means"][i],
                                        cov=self.dbn[evidence_var].pY["covariances"][i]) 
                for i in range(len(self.dbn[evidence_var].pY["weights"]))
            ])
            self.weights *= likelihoods

        # Normalize Weights
        self.weights += 1e-300  # Avoid zero weights
        self.weights /= np.sum(self.weights)

        # Resample particles
        self.resample()
