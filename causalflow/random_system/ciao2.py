from causallearn.graph.Dag import Dag
from causallearn.graph.Edge import Edge

from causallearn.utils.DAG2PAG import dag2pag
from causallearn.graph.Endpoint import Endpoint


def createDAG(link_assumptions, tau_max):
    nodes = []
    for n in link_assumptions.keys():
        for tau in range(tau_max + 1):
            nodes.append((n, -tau))
            
    dag = Dag(nodes)

    for t in link_assumptions.keys():
        for s, l, type in link_assumptions[t]:
            if type == '-->': dag.add_edge(Edge((s, l), (t, 0), Endpoint.TAIL, Endpoint.ARROW))
            elif type == '<->': dag.add_edge(Edge((s, l), (t, 0), Endpoint.ARROW, Endpoint.ARROW))
    return dag


# tau_max = 2
# DAG = {
#     'X_1': [('X_1', -1, '-->'), ('X_2', 0, '-->')],
#     'X_2': [],
#     'X_3': [('X_2', -1, '-->'), ('X_3', -1, '-->')],
#     'X_4': [('X_4', -1, '-->'), ('X_3', -2, '-->')],
# }

# tau_max = 1
# DAG = {
#     'X_1': [('X_2', -1, '-->')],
#     'X_2': [],
#     'X_3': [('X_2', -1, '-->')],
# }


# Create a DAG with the confounder U
dag = createDAG(DAG, tau_max)

# Convert the DAG to a PAG
pag = dag2pag(dag, islatent=[])
# pag = dag2pag(dag, islatent=[('X_2', -tau) for tau in range(0, tau_max + 1)])

# Convert PAG to dictionary representation
pag_dict = {}
for node in pag.nodes:
    pag_dict[node[0]] = []
for edge in pag.get_graph_edges():
    start, end = edge.node1, edge.node2
    if edge.endpoint1 == Endpoint.TAIL and edge.endpoint2 == Endpoint.ARROW:
        pag_dict[end[0]].append((start, '-->'))
    elif edge.endpoint1 == Endpoint.CIRCLE and edge.endpoint2 == Endpoint.ARROW:
        pag_dict[end[0]].append((start, 'o->'))
    elif edge.endpoint1 == Endpoint.TAIL and edge.endpoint2 == Endpoint.CIRCLE:
        pag_dict[end[0]].append((start, '--o'))
    elif edge.endpoint1 == Endpoint.ARROW and edge.endpoint2 == Endpoint.ARROW:
        pag_dict[end[0]].append((start, '<->'))
    elif edge.endpoint1 == Endpoint.CIRCLE and edge.endpoint2 == Endpoint.CIRCLE:
        pag_dict[end[0]].append((start, 'o-o'))
    else:
        raise ValueError('LinkType not considered!')

# Print the PAG dictionary
for key, value in pag_dict.items():
    print(f"{key}: {value}")
