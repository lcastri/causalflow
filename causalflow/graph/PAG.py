"""
This module provides the PAG class.

Classes:
    PAG: class for facilitating the handling and the creation of PAGs.
"""

from pgmpy.models import BayesianNetwork
from itertools import combinations
import copy
from collections import defaultdict

class PAG():
    """PAG class."""
    
    def __init__(self, dag, tau_max, latents) -> None:
        """
        Class constructor.

        Args:
            dag (DAG): DAG to convert.
            tau_max (int): max time lag.
            latents (list[str]): list of latent variables.

        Raises:
            ValueError: latent must be a string
        """
        if not isinstance(latents, list): raise ValueError('latents must be a list')
        self.link_assumptions = dag
        self.tau_max = tau_max
        self.latents = latents
        self.dSepSets = {}

        self.tsDAG = self.createDAG(self.link_assumptions, self.tau_max)
        
        self.pag = self.tsDAG2tsDPAG()
        
    
    def convert2Graph(self) -> dict:
        """
        Convert a PAG to a graph representation.

        Returns:
            dict: Graph representation of a PAG.
        """
        out = {t: {} for t in self.pag}
        for t in self.pag:
            for s in self.pag[t]:
                out[t][(s[0], s[1])] = s[2]
        return out
        
        
    @staticmethod
    def createDAG(link_assumptions, tau_max) -> BayesianNetwork:
        """
        Create a DAG represented by a Baysian Network.

        Args:
            link_assumptions (dict): DAG link assumptions.
            tau_max (int): max time lag.

        Raises:
            ValueError: source not well defined.

        Returns:
            BayesianNetwork: DAG represented by a Baysian Network.
        """
        BN = BayesianNetwork()
        BN.add_nodes_from([(t, -l) for t in link_assumptions.keys() for l in range(0, tau_max)])

        # Edges
        edges = []
        for t in link_assumptions.keys():
            for source in link_assumptions[t]:
                if len(source) == 0: continue
                elif len(source) == 2: s, l = source
                elif len(source) == 3: s, l, _ = source
                else: raise ValueError("Source not well defined")
                edges.append(((s, l), (t, 0)))
                # Add edges across time slices from -1 to -tau_max
                for lag in range(1, tau_max + 1):
                    if l - lag >= -tau_max:
                        edges.append(((s, l - lag), (t, -lag)))
        BN.add_edges_from(edges)
        return BN
    
    
    def alreadyChecked(self, source, target):
        """
        Check if a link has been already checked.

        Args:
            source (str): source node
            target (str): target node

        Returns:
            (bool, tuple): tuple containing if the link has been checked and, if so, their separation set. Otherwise None.
        """
        if (source, target) in self.dSepSets: return True, self.dSepSets[(source, target)]
        elif (target, source) in self.dSepSets: return True, self.dSepSets[(target, source)]
        elif ((source[0], source[1] - target[1]), (target[0], 0)) in self.dSepSets: return True, self.dSepSets[((source[0], source[1] - target[1]), (target[0], 0))]
        elif ((target[0], target[1] - source[1]), (source[0], 0)) in self.dSepSets: return True, self.dSepSets[((target[0], target[1] - source[1]), (source[0], 0))]
        return False, None

    
    def tsDAG2tsDPAG(self) -> dict:
        """
        Convert a DAG to a Time-series DPAG.

        Returns:
            dict: Time-series DPAG.
        """
        self.tsDPAG = {t: [(s[0], s[1], '-->') for s in self.link_assumptions[t] if s[0] not in self.latents] for t in self.link_assumptions.keys() if t not in self.latents}
        
        if len(self.latents) > 0:
            self.ambiguous_links = []
            
            # Separate nodes based on their time index
            time_zero_nodes = []
            other_nodes = []

            for node in self.tsDAG.nodes():
                if node[1] == 0:
                    time_zero_nodes.append(node)
                else:
                    other_nodes.append(node)
            
            for target in time_zero_nodes + other_nodes:
                print(f"Analysing target: {target}")
                if target[0] in self.latents: continue
                tmp = []
                for n in list(self.tsDAG.nodes()):
                    if n[0] != target[0] or n[1] != target[1]:
                        tmp.append(n)
                for p in list(self.tsDAG.predecessors(target)) + list(self.tsDAG.successors(target)):
                    if p in tmp: tmp.remove(p)
                for source in tmp:
                    alreadyChecked, d_sep = self.alreadyChecked(source, target)
                    if alreadyChecked:
                        print(f"\t- {source} ⊥ {target} | {d_sep} ALREADY CHECKED")
                    else:
                        areDsep, d_sep = self.find_d_separators(source, target, self.latents)
                        self.dSepSets[(source, target)] = d_sep
                        print(f"\t- {source} ⊥ {target} | {d_sep}")
                    if areDsep and any(node[0] in self.latents for node in d_sep):
                        if source[0] in self.latents or target[0] in self.latents: continue
                        if target[1] == 0:
                            print(f"\t- SPURIOUS LINK: ({source[0]}, {source[1]}) o-o ({target[0]}, {target[1]})")
                            if (source[0], source[1], 'o-o') not in self.tsDPAG[target[0]]: self.tsDPAG[target[0]].append((source[0], source[1], 'o-o'))
                            if (source, target, 'o-o') not in self.ambiguous_links: self.ambiguous_links.append((source, target, 'o-o'))
                        elif source[1] == 0:
                            print(f"\t- SPURIOUS LINK: ({target[0]}, {target[1]}) o-o ({source[0]}, {source[1]})")
                            if (target[0], target[1], 'o-o') not in self.tsDPAG[source[0]]: self.tsDPAG[source[0]].append((target[0], target[1], 'o-o'))
                            if (target, source, 'o-o') not in self.ambiguous_links: self.ambiguous_links.append((target, source, 'o-o'))
            
            
            print(f"--------------------------------------------------")
            print(f"    Bidirected link due to latent confounders     ")
            print(f"--------------------------------------------------")
            # *(1) Bidirected link between variables confounded by a latent variable  
            # *    if a link between them does not exist already
            confounders = self.find_latent_confounders()
            for confounded in copy.deepcopy(list(confounders.values())):
                for c1 in copy.deepcopy(confounded):
                    tmp = copy.deepcopy(confounded)
                    tmp.remove(c1)
                    for c2 in tmp:
                        if (c1, c2, 'o-o') in self.ambiguous_links:
                            self.update_link_type(c1, c2, '<->')
                            self.ambiguous_links.remove((c1, c2, 'o-o'))
                            print(f"\t- SPURIOUS LINK REMOVED: {c1} o-o {c2}")
                        elif (c2, c1, 'o-o') in self.ambiguous_links:
                            self.update_link_type(c1, c2, '<->')
                            self.ambiguous_links.remove((c2, c1, 'o-o'))
                            print(f"\t- SPURIOUS LINK REMOVED: {c2} o-o {c1}")
                    confounded.remove(c1)
                            
            print(f"--------------------------------------------------")
            print(f"              Collider orientation                ")
            print(f"--------------------------------------------------")
            # *(2) Identify and orient the colliders:
            # *    for any path X – Z – Y where there is no edge between
            # *    X and Y and, Z was never included in the conditioning set ==> X → Z ← Y collider
            colliders = self.find_colliders()
            for ambiguous_link in copy.deepcopy(self.ambiguous_links):
                source, target, linktype = ambiguous_link
                for parent1, collider, parent2 in colliders:
                    if collider == target and (parent1 == source or parent2 == source):
                        if not self.tsDAG.has_edge(parent1, parent2) and not self.tsDAG.has_edge(parent2, parent1):
                            self.update_link_type(parent1, target, '-->')
                            self.update_link_type(parent2, target, '-->')
                            self.ambiguous_links.remove(ambiguous_link)
                            break

            print(f"--------------------------------------------------")
            print(f"Non-collider orientation (orientation propagation)")
            print(f"--------------------------------------------------")
            # *(3) Orient the non-colliders edges (orientation propagation)
            # *    any edge Z – Y part of a partially directed path X → Z – Y,
            # *    where there is no edge between X and Y can be oriented as Z → Y
            for ambiguous_link in copy.deepcopy(self.ambiguous_links):
                triples = self.find_triples_containing_link(ambiguous_link)
                for triple in triples: self.update_link_type(triple[1], triple[2], '-->')
                
        # TODO: (3) Check if cycles are present
        
        else:
            print("No latent variable")
        
        return self.tsDPAG


    def find_colliders(self) -> list:
        """
        Find colliders.

        Returns:
            list: colliders.
        """
        colliders = []
        for node in self.tsDPAG.keys():
            parents = [(p[0], p[1]) for p in self.tsDPAG[node]]
            if len(parents) >= 2:
                for i in range(len(parents)):
                    for j in range(i + 1, len(parents)):
                        parent1 = parents[i]
                        parent2 = parents[j]
                        colliders.append((parent1, (node, 0), parent2))
        return colliders


    def update_link_type(self, parent, target, linktype):
        """
        Update link type.

        Args:
            parent (str): parent node.
            target (str): target node
            linktype (str): link type. E.g. --> or -?>.
        """
        for idx, link in enumerate(self.tsDPAG[target[0]]):
            if link[0] == parent[0] and link[1] == parent[1]:
                self.tsDPAG[target[0]][idx] = (link[0], link[1], linktype)
            
            
    def find_latent_confounders(self) -> dict:
        """
        Find latent confounders.

        Returns:
            dict: latent confounders.
        """
        confounders = {(latent, -t): list(self.tsDAG.successors((latent, -t))) for latent in self.latents for t in range(self.tau_max + 1) if len(list(self.tsDAG.successors((latent, -t)))) > 1}
        
        # Initialize a new dictionary to store unique edges
        shrinked_confounders = defaultdict(list)
        
        # Set to keep track of added edges without considering the time slice
        seen_edges = set()
        
        for key, value in confounders.items():
            # Normalize key by removing the time slice
            key_normalized = key[0]
            
            for v in value:
                # Normalize value by removing the time slice
                v_normalized = v[0]
                
                # Create a tuple of normalized edge
                edge = (key_normalized, v_normalized)
                
                # Check if edge or its reverse has been seen
                if edge not in seen_edges and (v_normalized, key_normalized) not in seen_edges:
                    # If not seen, add to unique edges and mark as seen
                    shrinked_confounders[key].append(v)
                    seen_edges.add(edge)
        
        return shrinked_confounders
    
                
    def find_d_separators(self, source, target):
        """
        Find D-Separation set.

        Args:
            source (str): source node.
            target (str): target node.

        Returns:
            (bool, set): (True, separation set) if source and target are d-separated. Otherwise (False, empty set). 
        """
        paths = self.find_all_paths(source, target)
               
        if len(paths) == 0: 
            return True, set()
        else:
            nodes = set()
            obs_nodes = set()
            for path in paths:
                path.remove(source)
                path.remove(target)
                for node in path:
                    nodes.add(node)
                    if node not in self.latents: obs_nodes.add(node)
                               
            for r in range(len(obs_nodes) + 1):
                for subset in combinations(obs_nodes, r):
                    subset_set = set(subset)
                    if not self.tsDAG.is_dconnected(source, target, subset_set):
                        return True, subset_set
                    
            for r in range(len(nodes) + 1):
                for subset in combinations(nodes, r):
                    subset_set = set(subset)
                    if not self.tsDAG.is_dconnected(source, target, subset_set):
                        return True, subset_set
            
        return False, set()

    def find_triples_containing_link(self, ambiguous_link) -> set:
        """
        Find all triples containing a link.

        Args:
            ambiguous_link (tuple): ambiguous_link

        Returns:
            set: triples containing the specified link.
        """
        pag = self.createDAG(self.tsDPAG, self.tau_max)

        source, target, _ = ambiguous_link
        triples = set()
        
        for n in pag.predecessors(source): 
            if n != target and not pag.has_edge(n, target) and not pag.has_edge(target, n): triples.add((n, source, target))
        for n in pag.predecessors(target): 
            if n != source and not pag.has_edge(n, source) and not pag.has_edge(source, n): triples.add((n, target, source))
        
        return triples
        
    
    # DFS to find all paths
    def find_all_paths(self, start, goal, path=[]) -> list:
        """
        Find all path from start to goal.

        Args:
            start (str): starting node.
            goal (str): goal node.
            path (list, optional): Found paths. Defaults to [].

        Returns:
            list: paths
        """
        path = path + [start]
        if start == goal:
            return [path]
        paths = []
        for node in self.tsDAG.successors(start):
            if node not in path:
                new_paths = self.find_all_paths(node, goal, path)
                for new_path in new_paths:
                    paths.append(new_path)
        for node in self.tsDAG.predecessors(start):
            if node not in path:
                new_paths = self.find_all_paths(node, goal, path)
                for new_path in new_paths:
                    paths.append(new_path)
        return paths

                    




# tau_max = 2

# DAG = {
#     'X_1': [('X_1', -1), ('X_2', -1), ('X_3', 0)],
#     'X_2': [],
#     'X_3': [('X_2', -1), ('X_4', -1)],
#     'X_4': [('X_4', -1)],
# }

# DAG = {
#     'X_1': [('X_2', 0), ('X_1', -1)],
#     'X_2': [],
#     'X_3': [('X_2', -1), ('X_3', -1)],
#     'X_4': [('X_4', -1), ('X_3', -2)],
# }

# p = PAG(DAG, tau_max, ['X_2'])
# print(p.pag)
