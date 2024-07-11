import copy
from pgmpy.models import BayesianNetwork





def convert_to_dpag(link_assumptions, latent):
    out = {t: [(s[0], s[1], '-->') for s in link_assumptions[t]] for t in link_assumptions.keys() }
    
    dag = createDAG(link_assumptions, tau_max)
    Hdag = createDAG(link_assumptions, tau_max, latent)
    
    # colliders = list(set([c for p1, c, p2 in find_colliders(dag)]))
    
    for target in dag.nodes():
        print(f"Target {target}")
        # tmp = [n for n in list(dag.nodes()) if n[0] != target[0] or n[1] != target[1]]
        tmp = []
        for n in list(dag.nodes()):
            if n[1] > target[1]: continue
            elif n[0] != target[0] or n[1] != target[1]:
                tmp.append(n)
        print(f"Others {tmp}")
        for source in tmp:
            if source not in link_assumptions[target[0]] and any(node[0] in latent for node in dag.minimal_dseparator(source, target)):
                out[target[0]].append((source[0], source[1], '?'))


def createDAG(link_assumptions, tau_max, latent = None):
    BN = BayesianNetwork()
    # Edges
    edges = []
    for t in link_assumptions.keys():
        for s, l in link_assumptions[t]:
            edges.append(((s, l), (t, 0)))
            
    # Net
    # tmp = BayesianNetwork(edges)
    if latent is not None:
        tmp = BayesianNetwork(edges, latents = set([(latent, -tau) for tau in range(0, tau_max + 1)]))
    else:
        tmp = BayesianNetwork(edges)
    return tmp


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