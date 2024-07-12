from pgmpy.models import BayesianNetwork

def convert_to_dpag(link_assumptions, latent):
    out = {t: [(s[0], s[1], '-->') for s in link_assumptions[t]] for t in link_assumptions.keys() }
    
    dag = createDAG(link_assumptions, tau_max)
    
    # colliders = list(set([c for p1, c, p2 in find_colliders(dag)]))
    
    for target in dag.nodes():
        if target[0] in latent: continue
        tmp = []
        for n in list(dag.nodes()):
            if n[1] > target[1]: continue
            elif n[0] != target[0] or n[1] != target[1]:
                tmp.append(n)
        for p in dag.predecessors(target):
            if p in tmp: tmp.remove(p)
        for p in dag.successors(target): 
            if p in tmp: tmp.remove(p)
        for source in tmp:
            if any(node[0] in latent for node in dag.minimal_dseparator(source, target)):
                out[target[0]].append((source[0], source[1], 'o-o'))
                
    # TODO: Spurious link identified. Now I need to orient it!
    # TODO: (1) Identify and orient the colliders:
    # TODO:     for any path X – Z – Y where there is no edge between
    # TODO:     X and Y and, Z was never included in the conditioning set ==> X → Z ← Y collider
    # TODO: (2) Orient the non-colliders edges (orientation propagation)
    # TODO:     any edge Z – Y part of a partially directed path X → Z – Y,
    # TODO:     where there is no edge between X and Y can be oriented as Z → Y
    # TODO: (3) Check if cycles are present

def createDAG(link_assumptions, tau_max, latent = None):
    BN = BayesianNetwork()
    BN.add_nodes_from([(t, -l) for t in link_assumptions.keys() for l in range(0, tau_max)])

    # Edges
    edges = []
    for t in link_assumptions.keys():
        for s, l in link_assumptions[t]:
            edges.append(((s, l), (t, 0)))
    BN.add_edges_from(edges)
            
            
    if latent is not None: BN.latents = latent
    return BN


def find_colliders(bayesian_model):
    colliders = []
    for node in bayesian_model.nodes:
        parents = list(bayesian_model.predecessors(node))
        if len(parents) >= 2:
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    parent1 = parents[i]
                    parent2 = parents[j]
                    if bayesian_model.has_edge(parent1, node) and bayesian_model.has_edge(parent2, node):
                        colliders.append((parent1, node, parent2))
    return colliders



# tau_max = 2
# LINK_ASSUMPTIONS = {
#     'X_1': [('X_1', -1, '-->'), ('X_2', 0, '-->')],
#     'X_2': [],
#     'X_3': [('X_2', -1, '-->'), ('X_3', -1, '-->')],
#     'X_4': [('X_4', -1, '-->'), ('X_3', -2, '-->')],
# }

tau_max = 2
LINK_ASSUMPTIONS = {
    'X_1': [('X_1', -1), ('X_2', -1), ('X_3', 0)],
    'X_2': [],
    'X_3': [('X_2', -1), ('X_4', -1)],
    'X_4': [('X_4', -1)],
}


convert_to_dpag(LINK_ASSUMPTIONS, 'X_2')
# print("COMPLETE DAG INDEPENDENCIES")
# print(dag.get_independencies())
# print()
# print("H DAG INDEPENDENCIES")
# print(Hdag.get_independencies())

# colliders = find_colliders(dag)


# Print colliders
# print("Colliders found:")
# for collider in colliders:
#     print(f"{collider[0]} -> {collider[1]} <- {collider[2]}")