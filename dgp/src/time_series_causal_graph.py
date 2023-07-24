"""
Copyright (c) 2020 by Impulse Innovations Ltd. Private and confidential. Part of the causaLens product.
"""

from copy import deepcopy
from itertools import product
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx
import numpy

from dgp.src.data_generation_configs import CausalGraphConfig


_DEFAULT_CAUSAL_GRAPH_CONFIG_VALUES = {
        0: {
            'max_lag': 1,
            'min_lag': 1,
            'num_targets': 1,
            'num_features': 3,
            'num_latent': 0,
            'prob_edge': 0.25,
            'max_parents_per_variable': 1,
            'max_target_parents': 1,
            'max_target_children': 0,
            'max_feature_parents': 2,
            'max_feature_children': 2,
            'max_latent_parents': 0,
            'max_latent_children': 0,
            'allow_latent_direct_target_cause': False,
            'allow_target_direct_target_cause': False,
            'prob_target_autoregressive': 0.0,
            'prob_feature_autoregressive': 0.0,
            'prob_latent_autoregressive': 0.0,
            'prob_noise_autoregressive': 0.0
        },
        10: {
            'max_lag': 2,
            'min_lag': 1,
            'num_targets': 1,
            'num_features': 5,
            'num_latent': 0,
            'prob_edge': 0.3,
            'max_parents_per_variable': 1,
            'max_target_parents': 2,
            'max_target_children': 0,
            'max_feature_parents': 3,
            'max_feature_children': 3,
            'max_latent_parents': 0,
            'max_latent_children': 0,
            'allow_latent_direct_target_cause': False,
            'allow_target_direct_target_cause': False,
            'prob_target_autoregressive': 0.0,
            'prob_feature_autoregressive': 0.0,
            'prob_latent_autoregressive': 0.0,
            'prob_noise_autoregressive': 0.0
        },
        20: {
            'max_lag': 4,
            'min_lag': 1,
            'num_targets': 1,
            'num_features': 10,
            'num_latent': 3,
            'prob_edge': 0.4,
            'max_parents_per_variable': 2,
            'max_target_parents': 3,
            'max_target_children': 1,
            'max_feature_parents': 4,
            'max_feature_children': 4,
            'max_latent_parents': 3,
            'max_latent_children': 3,
            'allow_latent_direct_target_cause': False,
            'allow_target_direct_target_cause': False,
            'prob_target_autoregressive': 0.3,
            'prob_feature_autoregressive': 0.3,
            'prob_latent_autoregressive': 0.3,
            'prob_noise_autoregressive': 0.1
        },
        30: {
            'max_lag': 6,
            'min_lag': 1,
            'num_targets': 1,
            'num_features': 50,
            'num_latent': 15,
            'prob_edge': 0.5,
            'max_parents_per_variable': 3,
            'max_target_parents': 5,
            'max_target_children': 2,
            'max_feature_parents': 8,
            'max_feature_children': 8,
            'max_latent_parents': 5,
            'max_latent_children': 5,
            'allow_latent_direct_target_cause': True,
            'allow_target_direct_target_cause': True,
            'prob_target_autoregressive': 0.4,
            'prob_feature_autoregressive': 0.4,
            'prob_latent_autoregressive': 0.4,
            'prob_noise_autoregressive': 0.25
        }
    }


class TimeSeriesCausalGraph:
    """
    This class generates a valid time series causal graph, represented as a directed acyclic graph (DAG), from the
    provided CausalGraphConfig object.
    """

    def __init__(self, graph_config: CausalGraphConfig, rng: numpy.random.Generator):
        """
        A time series causal graph represented as a directed acyclic graph (DAG).

        :param graph_config: The CausalGraphConfig object used to parameterise the process to randomly generate
            the time series causal graph.
        :param rng: A random number Generator.
        """
        self.graph_config = deepcopy(graph_config)
        self._complete_causal_graph_config()
        self._rng = rng
        self.causal_graph: networkx.DiGraph = self._generate_causal_graph()

    def get_target_nodes(self) -> List[str]:
        """ Returns a list of all target nodes. """
        return [node for node in self.causal_graph.nodes if node.startswith('Y')]

    def get_feature_nodes(self) -> List[str]:
        """ Returns a list of all feature nodes. """
        return [node for node in self.causal_graph.nodes if node.startswith('X')]

    def get_latent_nodes(self) -> List[str]:
        """ Returns a list of all latent nodes. """
        return [node for node in self.causal_graph.nodes if node.startswith('U')]

    def get_noise_nodes(self) -> List[str]:
        """ Returns a list of all noise nodes. """
        return [node for node in self.causal_graph.nodes if node.startswith('S')]

    def get_unique_variables(self, include_noise: bool = False) -> List[str]:
        """
        Returns a list of all unique variables. This is not the same as the list of all nodes as each variable
        (e.g., Y1) may have multiple nodes (e.g., Y1 at time t and time t-1 would be two nodes for the same variable).

        :param include_noise: Boolean to control if noise variables should be included. Default is False.
        :return: Returns a list of all unique variables.
        """
        if include_noise:
            return sorted({node.split('_')[0] for node in self.causal_graph.nodes})
        else:
            return sorted({node.split('_')[0] for node in self.causal_graph.nodes if not node.startswith('S')})

    def get_unique_noise_variables(self) -> List[str]:
        """ Returns a list of all unique noise variables. """
        return sorted({node.split('_')[0] for node in self.causal_graph.nodes if node.startswith('S')})

    def get_topological_order_of_t_nodes(self) -> List[str]:
        """ Returns a list of all nodes (excluding noise) at time t in their topological order. """
        # Ignore noise nodes as those are either IID or auto-regressive.
        unique_t_nodes = [node + '_t' for node in self.get_unique_variables(include_noise=False)]
        if self.graph_config.min_lag > 0:
            # Do not care about topological order as graph does not allow for instantaneous effects.
            return unique_t_nodes
        else:
            return [node for node in networkx.topological_sort(self.causal_graph) if node in unique_t_nodes]

    def get_all_var_nodes(self, var: str) -> List[str]:
        """ Return list of all the nodes for the specified variable name."""
        current_node = var + '_t'
        ret_val = []
        while self.get_node_lag(current_node) <= self.graph_config.max_lag:
            ret_val.append(current_node)
            current_node = self.lag_node(current_node)
        return ret_val

    def get_parents(self, node: str) -> List[str]:
        """ Returns a list of all the parent nodes for the specified node. """
        return list(self.causal_graph.predecessors(node))

    def get_children(self, node: str) -> List[str]:
        """ Returns a list of all the children nodes for the specified node. """
        return list(self.causal_graph.successors(node))

    def get_number_of_parents(self, node: str) -> int:
        """ Returns the number of parents for the specified node. """
        return len(list(self.causal_graph.predecessors(node)))

    def get_number_of_children(self, node: str) -> int:
        """ Returns the number of children for the specified node. """
        return len(list(self.causal_graph.successors(node)))

    def get_number_of_parents_of_variable(self, node: str, var_name: str) -> int:
        """
        Returns the number of parents of the specified node for the specified variable. For example, if the specified
        node is Y1 at time t, the specified variable is X1, and both X1 at time t and at time t-2 are parents of Y1,
        then the returned value would be 2.

        :param node: The node for which the number of parents are to be calculated.
        :param var_name: The specific variable to count how many times it is a parent of the specified node.
        :return: Returns the number of parents of the specified node for the specified variable.
        """
        return len([parent for parent in self.causal_graph.predecessors(node) if parent.startswith(var_name)])

    def generate_summary_graph(self, include_latent_nodes: bool = False, include_autoregressive_edges: bool = False,
                               include_indirect_causes: bool = False, include_noise: bool = False) -> networkx.DiGraph:
        """
        This method generates the summary graph for the time series causal graph. The time series causal graph is a
        valid DAG, but the summary graph is a directed graph that is not guaranteed to be acyclic. Section 10.1 of
        "Peters, J., Janzing, D. and Schölkopf, B., 2017. Elements of causal inference. The MIT Press." provides a
        general definition of a summary graph. This is the same definition as that in T2 of Section 2 of the SyPI paper:
        "Mastakouri, A.A., Schölkopf, B. and Janzing, D., 2020. Necessary and sufficient conditions for causal feature
        selection in time series with latent common causes. arXiv preprint arXiv:2005.08543."

        This method provides the capability to return a summary graph that differs slightly from their definition. As
        their summary graph cannot include self loops as they specify nodes j /= k, but we may want to see those edges
        for other purposes. As such parameters are provided to allow for the inclusion of self loops, latent nodes,
        and/or noise nodes. The default parameter values will return the summary graph per the author's definition.

        :param include_latent_nodes: A boolean to specify whether or not to include latent nodes in the summary graph.
        :param include_autoregressive_edges: A boolean to specify whether or not to include self loops in the summary
            graph.
        :param include_indirect_causes: A boolean to specify whether to include links via indirect causes when latent
            nodes are not included. For example, if the following links exist in the full graph: x1 -> u1 -> x2, and
            latent nodes are not included in the summary graph, then x1 -> x2 edge will be included when this parameter
            is set to True.
        :param include_noise: A boolean to specify whether or not to include noise nodes in the summary graph.
        :return: A directed graph representing the summary graph of the time series causal graph. There is no guarantee
            that this is a directed acyclic graph.
        """

        # Get all unique variables, i.e., ignoring lags, in the causal graph.
        summary_graph_nodes = self.get_unique_variables(include_noise=include_noise)

        # Remove latent nodes if they are not specified to be included.
        if not include_latent_nodes:
            summary_graph_nodes = [node for node in summary_graph_nodes if not node.startswith('U')]

        # Instantiate empty directed graph with the summary graph nodes.
        summary_graph = networkx.empty_graph(n=summary_graph_nodes, create_using=networkx.DiGraph)

        # Iterate through each node at time t and add edges in summary graph from appropriate parents.
        for node in summary_graph_nodes:
            unique_variable_parents = {parent.split('_')[0] for parent in self.get_parents(node + '_t')}

            # Remove any nodes that are not in list of summary graph nodes. This is likely the case as each node will
            # often have a noise node as a parent. However, if noise nodes are not to be included, they must be ignored.
            unique_variable_parents = unique_variable_parents.intersection(summary_graph_nodes)

            # Remove node from parents if self loops are not specified to be included.
            if not include_autoregressive_edges:
                unique_variable_parents.discard(node)  # Removes node if it is in set; otherwise, does nothing.

            # Add edge for each parent, ignoring self loop if not specified to be included.
            summary_graph.add_edges_from((parent, node) for parent in unique_variable_parents)

        # Check to see if indirect links need to be added.
        if not include_latent_nodes and include_indirect_causes:
            # Get summary graph with latent nodes for comparison.
            summary_graph_with_latents = self.generate_summary_graph(
                include_latent_nodes=True, include_autoregressive_edges=include_autoregressive_edges,
                include_indirect_causes=False, include_noise=include_noise
            )

            # Compare nodes in summary graph to see if they have path in the summary graph with latent nodes.
            for source, target in product(summary_graph_nodes, summary_graph_nodes):
                # Skip to next step if directed edge already exists between source and target nodes, there is no path
                # in the summary graph with latents or if looking at same node as self loops cannot be influenced by
                # unobserved (latent) variables.
                if (source == target or summary_graph.has_edge(source, target) or
                        not networkx.has_path(summary_graph_with_latents, source, target)):
                    continue

                # Iterate over all paths between source and target to see if path includes only latent nodes.
                for path in networkx.all_simple_paths(summary_graph_with_latents, source, target):
                    if len(path) > 2 and all(self.is_latent_node(node) for node in path[1:-1]):
                        # Add edge in summary graph as the path is through only latent nodes so it is indirect cause.
                        summary_graph.add_edge(source, target, is_indirect=True)
                        break  # Exit for-loop as we don't need to check any more paths as we have found indirect cause.

        return summary_graph

    @staticmethod
    def get_var_and_lag(node: str) -> Tuple[str, int]:
        """ Returns the variable name and lag for the specified node. """
        return TimeSeriesCausalGraph.get_node_var(node), TimeSeriesCausalGraph.get_node_lag(node)

    @staticmethod
    def get_node_var(node: str) -> str:
        """ Returns the variable name for the specified node. """
        return node.split('_')[0]

    @staticmethod
    def get_node_lag(node: str) -> int:
        """ Returns the lag for the specified node. """
        if node.endswith('t'):
            return 0
        else:
            return int(node.split('-')[-1])

    @staticmethod
    def lag_node(node: str, lag: int = 1) -> str:
        """ Returns the node name for the specified node lagged by the specified lag value. """
        assert isinstance(lag, int) and lag > 0, f'lag must be a positive integer; got {lag}.'
        if node.endswith('t'):
            return node + f'-{lag}'
        else:
            node_prefix, current_lag = node.split('-')
            new_lag = int(current_lag) + lag
            return node_prefix + f'-{new_lag}'

    @staticmethod
    def is_target_node(node: str) -> bool:
        """ Returns True if the specified node is a target variable. """
        return node.startswith('Y')

    @staticmethod
    def is_feature_node(node: str) -> bool:
        """ Returns True if the specified node is a feature variable. """
        return node.startswith('X')

    @staticmethod
    def is_latent_node(node: str) -> bool:
        """ Returns True if the specified node is a latent variable. """
        return node.startswith('U')

    @staticmethod
    def is_noise_node(node: str) -> bool:
        """ Returns True if the specified node is a noise variable. """
        return node.startswith('S')

    @staticmethod
    def number_of_parents(causal_graph: networkx.DiGraph, node: str) -> int:
        """ Returns the number of parents for the specified node in the specified graph. """
        return len(list(causal_graph.predecessors(node)))

    @staticmethod
    def number_of_children(causal_graph: networkx.DiGraph, node: str) -> int:
        """ Returns the number of children for the specified node in the specified graph. """
        return len(list(causal_graph.successors(node)))

    @staticmethod
    def number_of_parents_of_variable(causal_graph: networkx.DiGraph, node: str, var_name: str) -> int:
        """
        Returns the number of parents of the specified node for the specified variable. For example, if the specified
        node is Y1 at time t, the specified variable is X1, and both X1 at time t and at time t-2 are parents of Y1,
        then the returned value would be 2.

        :param causal_graph: The causal graph to perform the calculation on.
        :param node: The node for which the number of parents are to be calculated.
        :param var_name: The specific variable to count how many times it is a parent of the specified node.
        :return: Returns the number of parents of the specified node for the specified variable.
        """

        return len([parent for parent in causal_graph.predecessors(node) if parent.startswith(var_name)])

    def display_graph(self, include_noise: bool = False):
        """
        Helper method to plot the causal graph using matplotlib.

        :param include_noise: Boolean to specify if noise nodes should be displayed. Default is False.
        """

        x_positions = numpy.linspace(1.0, 0.0, self.graph_config.max_lag + 1)
        y_positions = numpy.linspace(1.0, 0.0, len(self.get_unique_variables(include_noise=include_noise)))

        # Use perceptually uniform sequential or qualitative colormap.
        # See https://matplotlib.org/tutorials/colors/colormaps.html
        color_map = {'Y': 0.0, 'X': 0.33, 'U': 0.67, 'S': 1.0}

        if include_noise and self.graph_config.include_noise:
            node_list = self.causal_graph.nodes  # All nodes.
            edge_list = self.causal_graph.edges  # All edges.
            color_vals = [color_map.get(node[0], 0.5) for node in self.causal_graph.nodes]

            # Define display order.
            display_order = [
                [node for node in self.causal_graph.nodes if node.startswith(f'Sy{i}')] +
                [node for node in self.causal_graph.nodes if node.startswith(f'Y{i}')]
                for i in range(1, self.graph_config.num_targets + 1)
            ]
            display_order += [
                [node for node in self.causal_graph.nodes if node.startswith(f'Sx{i}')] +
                [node for node in self.causal_graph.nodes if node.startswith(f'X{i}')]
                for i in range(1, self.graph_config.num_features + 1)
            ]
            display_order += [
                [node for node in self.causal_graph.nodes if node.startswith(f'Su{i}')] +
                [node for node in self.causal_graph.nodes if node.startswith(f'U{i}')]
                for i in range(1, self.graph_config.num_latent + 1)
            ]
            # Flatten display order and set (x,y) positions.
            display_order = [item for sublist in display_order for item in sublist]

        else:
            node_list = [node for node in self.causal_graph.nodes if not node.startswith('S')]  # No noise.
            # No edges originating from noise.
            edge_list = [edge for edge in self.causal_graph.edges if not edge[0].startswith('S')]
            color_vals = [color_map.get(node[0], 0.5) for node in self.causal_graph.nodes if not node.startswith('S')]

            display_order = self.get_target_nodes() + self.get_feature_nodes() + self.get_latent_nodes()

        # Set (x,y) positions for each node.
        positions = dict(zip(display_order, [numpy.array([x, y]) for y in y_positions for x in x_positions]))

        # Draw graph. NetworkX draw fails if pos does not include all nodes in labels. As such we must provide labels
        # for the nodes we want to display.
        plt.figure()
        networkx.draw(self.causal_graph, pos=positions, with_labels=True, nodelist=node_list, edgelist=edge_list,
                      node_size=550, font_size=8, node_color=color_vals, cmap=plt.get_cmap('Set2'),
                      labels={node: node for node in display_order})

    def _generate_causal_graph(self) -> networkx.DiGraph:
        """
        This method builds the time series causal graph given the completed CausalGraphConfig object and the random
        number Generator.

        :return: Returns a networkx DiGraph capturing the completed time series causal graph as a valid DAG.
        """

        # Determine the total number of nodes.
        max_lag = self.graph_config.max_lag
        num_targets = self.graph_config.num_targets
        num_features = self.graph_config.num_features
        num_latent = self.graph_config.num_latent
        num_nodes = (1 + self.graph_config.include_noise) * (num_targets + num_features + num_latent) * (1 + max_lag)

        # Check at least one node.
        assert num_nodes > 0, 'Should have at least one node in the causal graph.'

        # The causal graph must be a directed acyclic graph (DAG).

        # Instantiate empty adjacency matrix to describe DAG and list of strings describing nodes.
        adjacency_matrix = numpy.zeros(shape=(num_nodes, num_nodes), dtype=int)
        nodes = (
                [f'Y{i}_t-{t}'.replace('-0', '') for i in range(1, num_targets + 1) for t in range(max_lag + 1)] +
                [f'X{i}_t-{t}'.replace('-0', '') for i in range(1, num_features + 1) for t in range(max_lag + 1)] +
                [f'U{i}_t-{t}'.replace('-0', '') for i in range(1, num_latent + 1) for t in range(max_lag + 1)]
        )
        if self.graph_config.include_noise:
            nodes += (
                    [f'Sy{i}_t-{t}'.replace('-0', '') for i in range(1, num_targets + 1) for t in range(max_lag + 1)] +
                    [f'Sx{i}_t-{t}'.replace('-0', '') for i in range(1, num_features + 1) for t in range(max_lag + 1)] +
                    [f'Su{i}_t-{t}'.replace('-0', '') for i in range(1, num_latent + 1) for t in range(max_lag + 1)]
            )

        # Sanity check.
        assert len(nodes) == num_nodes, f'Number of nodes does not match expected number ({num_nodes}).'

        # Add autoregressive edges.
        if max_lag > 0:
            # Add autoregressive edges for target variables.
            if self.graph_config.prob_target_autoregressive > 0.0:
                # Iterate over each target variable and determine if there is an autoregressive link by sampling from a
                # Bernoulli distribution parameterised by `prob_target_autoregressive` in the config.
                for target_idx in range(num_targets):
                    add_autoregressive_edge = self._rng.random() < self.graph_config.prob_target_autoregressive
                    if add_autoregressive_edge:
                        # Add link between all consecutive (in time) nodes.
                        for t in range(1, max_lag + 1):
                            source_idx = target_idx * (max_lag + 1) + t
                            adjacency_matrix[source_idx][source_idx - 1] = 1
                    # No else needed as draw from Bernoulli distribution was False.
            # No else needed as autoregressive edges for target variables are not allowed.

            # Add autoregressive edges for feature variables.
            if self.graph_config.prob_feature_autoregressive > 0.0:
                # Iterate over each feature variable and determine if there is an autoregressive link by sampling from a
                # Bernoulli distribution parameterised by `prob_feature_autoregressive` in the config.
                for feature_idx in range(num_features):
                    add_autoregressive_edge = self._rng.random() < self.graph_config.prob_feature_autoregressive
                    if add_autoregressive_edge:
                        # Add link between all consecutive (in time) nodes.
                        for t in range(1, max_lag + 1):
                            source_idx = (num_targets + feature_idx) * (max_lag + 1) + t
                            adjacency_matrix[source_idx][source_idx - 1] = 1
                    # No else needed as draw from Bernoulli distribution was False.
            # No else needed as autoregressive edges for feature variables are not allowed.

            # Add autoregressive edges for latent variables.
            if self.graph_config.prob_latent_autoregressive > 0.0:
                # Iterate over each latent variable and determine if there is an autoregressive link by sampling from a
                # Bernoulli distribution parameterised by `prob_latent_autoregressive` in the config.
                for latent_idx in range(num_latent):
                    add_autoregressive_edge = self._rng.random() < self.graph_config.prob_latent_autoregressive
                    if add_autoregressive_edge:
                        # Add link between all consecutive (in time) nodes.
                        for t in range(1, max_lag + 1):
                            source_idx = (num_targets + num_features + latent_idx) * (max_lag + 1) + t
                            adjacency_matrix[source_idx][source_idx - 1] = 1
                    # No else needed as draw from Bernoulli distribution was False.
            # No else needed as autoregressive edges for latent variables are not allowed.

            # Add autoregressive edges for noise variables.
            if self.graph_config.include_noise and self.graph_config.prob_noise_autoregressive > 0.0:
                # Iterate over each noise variable and determine if there is an autoregressive link by sampling from a
                # Bernoulli distribution parameterised by `prob_noise_autoregressive` in the config.
                for noise_idx in range(num_targets + num_features + num_latent):
                    add_autoregressive_edge = self._rng.random() < self.graph_config.prob_noise_autoregressive
                    if add_autoregressive_edge:
                        # Add link between all consecutive (in time) nodes.
                        for t in range(1, max_lag + 1):
                            source_idx = (num_targets + num_features + num_latent + noise_idx) * (max_lag + 1) + t
                            adjacency_matrix[source_idx][source_idx - 1] = 1
                    # No else needed as draw from Bernoulli distribution was False.
            # No else needed as autoregressive edges for noise variables are not allowed.

        # No else needed as max_lag is 0. Therefore, autoregressive edges are not possible.

        # Generate the DAG from the adjacency matrix and label nodes.
        
        dag = networkx.DiGraph(adjacency_matrix, parallel_edges=False, create_using=networkx.DiGraph)
        # dag = networkx.from_numpy_matrix(adjacency_matrix, parallel_edges=False, create_using=networkx.DiGraph)
        networkx.relabel_nodes(dag, {i: nodes[i] for i in range(len(nodes))}, copy=False)

        # Iterator over every node at time t and determine parents.
        t_nodes = [node for node in dag.nodes if not node.startswith('S') and node.endswith('t')]
        parent_thresholds = {
            'Y': self.graph_config.max_target_parents,
            'X': self.graph_config.max_feature_parents,
            'U': self.graph_config.max_latent_parents
        }
        children_thresholds = {
            'Y': self.graph_config.max_target_children,
            'X': self.graph_config.max_feature_children,
            'U': self.graph_config.max_latent_children
        }
        edge_probability = {
            'Y': self.graph_config.prob_target_parent if self.graph_config.prob_target_parent is not None
            else self.graph_config.prob_edge,
            'X': self.graph_config.prob_feature_parent if self.graph_config.prob_feature_parent is not None
            else self.graph_config.prob_edge,
            'U': self.graph_config.prob_latent_parent if self.graph_config.prob_latent_parent is not None
            else self.graph_config.prob_edge
        }
        # Possible parents are all those nodes with lag between max lag and time delta and not equal to itself.
        for t_node in t_nodes:
            possible_parents = [node for node in dag.nodes if (not node.startswith('S') and
                                                               not node.startswith(self.get_node_var(t_node)))]

            if self.graph_config.min_lag > 0:
                # Reduce possible parents to fall within min_lag.
                possible_parents = [
                    node for node in possible_parents if self.get_node_lag(node) >= self.graph_config.min_lag
                ]
            else:
                # Ensure no possible parents are descendants of current node, which would invalidate the DAG by
                # introducing a cycle. This can only happen when instantaneous effects are allowed, e.g., `min_lag` = 0.
                possible_parents = [node for node in possible_parents if node not in networkx.descendants(dag, t_node)]

            # Further reduce possible parents as targets may not be allowed to be driven directly by other targets or
            # driven directly by latent variables.
            if self.is_target_node(t_node):
                if not self.graph_config.allow_latent_direct_target_cause:
                    possible_parents = [node for node in possible_parents if not self.is_latent_node(node)]
                if not self.graph_config.allow_target_direct_target_cause:
                    possible_parents = [node for node in possible_parents if not self.is_target_node(node)]
            # No else needed as current node is not a target variable.

            # Finally, with list of possible parents, randomly permute list so each node does not follow the same order.
            self._rng.shuffle(possible_parents)

            # Determine maximum possible number of parents and edge probability given current node type.
            max_num_parents = parent_thresholds[t_node[0]]
            prob_edge = edge_probability[t_node[0]]

            # Iterate over each possible parent and draw possible edge up to max number of parents.
            # Note: Currently an autoregressive link counts as one parent.
            # Check if t_node has reached its max for number of parents and possible parents remaining.
            while self.number_of_parents(dag, t_node) < max_num_parents and len(possible_parents) > 0:
                # Use first possible parent as current node.
                p_node = possible_parents.pop(0)

                # Determine maximum number of children allowed for current possible parent node.
                max_num_children = children_thresholds[p_node[0]]

                # Draw from Bernoulli distribution parameterised by the config to see if we should add edge and
                # confirm p_node is not at its max number of children.
                if self._rng.random() < prob_edge and self.number_of_children(dag, p_node) < max_num_children:
                    # Add edge.
                    dag.add_edge(p_node, t_node)
                    # Iterate through each lag until we hit max_lag.
                    current_lag = self.get_node_lag(p_node)
                    p_node_lagged = p_node
                    t_node_lagged = t_node
                    while current_lag < max_lag:
                        p_node_lagged = self.lag_node(p_node_lagged)
                        t_node_lagged = self.lag_node(t_node_lagged)
                        dag.add_edge(p_node_lagged, t_node_lagged)
                        current_lag = self.get_node_lag(p_node_lagged)

                # Check `max_parents_per_variable` to see if we should remove other lags of current variable in p_node
                # from list of possible parents.
                p_node_var = self.get_node_var(p_node)
                if (self.number_of_parents_of_variable(causal_graph=dag, node=t_node, var_name=p_node_var) >=
                        self.graph_config.max_parents_per_variable):
                    # Remove any other instances of the same variable from possible parents.
                    possible_parents = [node for node in possible_parents if not node.startswith(p_node_var)]
                # No else needed as this max value has not been exceeded.

        # Add links for noise. Easier to do with networkx than indexing the adjacency matrix.
        # Note, this will increase number of parents for each node, but we do not want to count noise as contributing
        # to max parents as the graph either includes noise or not. So every node gets a corresponding noise node if
        # include_noise is set to True.
        if self.graph_config.include_noise:
            target_noise_edges = [(f'Sy{i}_t-{t}'.replace('-0', ''), f'Y{i}_t-{t}'.replace('-0', ''))
                                  for i in range(1, num_targets + 1) for t in range(max_lag + 1)]
            dag.add_edges_from(target_noise_edges)

            feature_noise_edges = [(f'Sx{i}_t-{t}'.replace('-0', ''), f'X{i}_t-{t}'.replace('-0', ''))
                                   for i in range(1, num_features + 1) for t in range(max_lag + 1)]
            dag.add_edges_from(feature_noise_edges)

            latent_noise_edges = [(f'Su{i}_t-{t}'.replace('-0', ''), f'U{i}_t-{t}'.replace('-0', ''))
                                  for i in range(1, num_latent + 1) for t in range(max_lag + 1)]
            dag.add_edges_from(latent_noise_edges)

        # Finally, confirm graph is a valid DAG.
        assert networkx.is_directed_acyclic_graph(dag), 'Generated causal graph is not a valid directed acyclic graph.'

        return dag

    def _complete_causal_graph_config(self):
        """
        This method updates the causal graph configuration with default values for any unspecified parameters based off
        the specified complexity parameter value.
        """

        # Validate complexity and set default values if parameters are not provided.
        assert self.graph_config.graph_complexity in [0, 10, 20, 30], \
            f'graph_complexity must be one of 0, 10, 20, or 30. Got {self.graph_config.graph_complexity}.'

        complexity = self.graph_config.graph_complexity

        #  Set default values if parameters are not provided.
        if self.graph_config.max_lag is None:
            self.graph_config.max_lag = _DEFAULT_CAUSAL_GRAPH_CONFIG_VALUES[complexity]['max_lag']

        if self.graph_config.min_lag is None:
            self.graph_config.min_lag = _DEFAULT_CAUSAL_GRAPH_CONFIG_VALUES[complexity]['min_lag']

        if self.graph_config.num_targets is None:
            self.graph_config.num_targets = _DEFAULT_CAUSAL_GRAPH_CONFIG_VALUES[complexity]['num_targets']

        if self.graph_config.num_features is None:
            self.graph_config.num_features = _DEFAULT_CAUSAL_GRAPH_CONFIG_VALUES[complexity]['num_features']

        if self.graph_config.num_latent is None:
            self.graph_config.num_latent = _DEFAULT_CAUSAL_GRAPH_CONFIG_VALUES[complexity]['num_latent']

        if self.graph_config.prob_edge is None:
            self.graph_config.prob_edge = _DEFAULT_CAUSAL_GRAPH_CONFIG_VALUES[complexity]['prob_edge']
        # Do not set default values for prob_target_parent, prob_feature_parent, prob_latent_parent as prob_edge is
        # used if those are undefined.

        if self.graph_config.max_parents_per_variable is None:
            self.graph_config.max_parents_per_variable = _DEFAULT_CAUSAL_GRAPH_CONFIG_VALUES[complexity][
                'max_parents_per_variable'
            ]

        if self.graph_config.max_target_parents is None:
            self.graph_config.max_target_parents = _DEFAULT_CAUSAL_GRAPH_CONFIG_VALUES[complexity]['max_target_parents']

        if self.graph_config.max_target_children is None:
            self.graph_config.max_target_children = _DEFAULT_CAUSAL_GRAPH_CONFIG_VALUES[complexity][
                'max_target_children'
            ]

        if self.graph_config.max_feature_parents is None:
            self.graph_config.max_feature_parents = _DEFAULT_CAUSAL_GRAPH_CONFIG_VALUES[complexity][
                'max_feature_parents'
            ]

        if self.graph_config.max_feature_children is None:
            self.graph_config.max_feature_children = _DEFAULT_CAUSAL_GRAPH_CONFIG_VALUES[complexity][
                'max_feature_children'
            ]

        if self.graph_config.max_latent_parents is None:
            self.graph_config.max_latent_parents = _DEFAULT_CAUSAL_GRAPH_CONFIG_VALUES[complexity]['max_latent_parents']

        if self.graph_config.max_latent_children is None:
            self.graph_config.max_latent_children = _DEFAULT_CAUSAL_GRAPH_CONFIG_VALUES[complexity][
                'max_latent_children'
            ]

        if self.graph_config.allow_latent_direct_target_cause is None:
            self.graph_config.allow_latent_direct_target_cause = _DEFAULT_CAUSAL_GRAPH_CONFIG_VALUES[complexity][
                'allow_latent_direct_target_cause'
            ]

        if self.graph_config.allow_target_direct_target_cause is None:
            self.graph_config.allow_target_direct_target_cause = _DEFAULT_CAUSAL_GRAPH_CONFIG_VALUES[complexity][
                'allow_target_direct_target_cause'
            ]

        if self.graph_config.prob_target_autoregressive is None:
            self.graph_config.prob_target_autoregressive = _DEFAULT_CAUSAL_GRAPH_CONFIG_VALUES[complexity][
                'prob_target_autoregressive'
            ]

        if self.graph_config.prob_feature_autoregressive is None:
            self.graph_config.prob_feature_autoregressive = _DEFAULT_CAUSAL_GRAPH_CONFIG_VALUES[complexity][
                'prob_feature_autoregressive'
            ]

        if self.graph_config.prob_latent_autoregressive is None:
            self.graph_config.prob_latent_autoregressive = _DEFAULT_CAUSAL_GRAPH_CONFIG_VALUES[complexity][
                'prob_latent_autoregressive'
            ]

        if self.graph_config.prob_noise_autoregressive is None:
            self.graph_config.prob_noise_autoregressive = _DEFAULT_CAUSAL_GRAPH_CONFIG_VALUES[complexity][
                'prob_noise_autoregressive'
            ]

        # Confirm max_lag and min_lag are not in contention.
        assert self.graph_config.min_lag <= self.graph_config.max_lag, (
            f'min_lag must be less than or equal to max_lag. Currently, min_lag is {self.graph_config.min_lag}'
            f' and max_lag is{self.graph_config.max_lag}.')
