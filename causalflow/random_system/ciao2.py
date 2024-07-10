# import numpy as np
# from causallearn.graph.Dag import Dag
# from causallearn.graph.Edge import Edge
# from causallearn.graph.Endpoint import Endpoint
# from causallearn.utils.DAG2PAG import dag2pag

# min_lag = 0
# max_lag = 0
# DAG = {
#     'X_1': [('X_2', 0, '-->')],
#     'X_2': [],
#     'X_3': [('X_2', 0, '-->')],
# }
# # DAG = {
# #     'X_1': [('X_1', -1, '-->'), ('X_2', -1, '-->'), ('X_3', 0, '-->')],
# #     'X_2': [],
# #     'X_3': [('X_4', -1, '-->'), ('X_2', -1, '-->')],
# #     'X_4': [('X_4', -1, '-->')]
# # }

# nodes = []
# for node in DAG.keys():
#     for l in range(max_lag-min_lag+1):
#         nodes.append((node, -l))
    
# # Define a simple DAG
# dag = Dag(nodes=nodes)
# for node in DAG.keys():
#     for source in DAG[node]:
#         dag.add_edge(Edge((source[0], source[1]), (node, 0), Endpoint.TAIL, Endpoint.ARROW))

# # Convert the DAG to a PAG
# pag = dag2pag(dag, [('X_2', -l) for l in range(max_lag-min_lag+1)])

# # Convert PAG to dictionary representation
# pag_dict = {}
# for node in pag.nodes:
#     pag_dict[node[0]] = []
# for edge in pag.get_graph_edges():
#     start, end = edge.node1, edge.node2
#     if edge.endpoint1 == Endpoint.TAIL and edge.endpoint2 == Endpoint.ARROW:
#         pag_dict[end[0]].append((start[0], start[1], '-->'))
#     elif edge.endpoint1 == Endpoint.CIRCLE and edge.endpoint2 == Endpoint.ARROW:
#         pag_dict[end[0]].append((start[0], start[1], 'o->'))
#     elif edge.endpoint1 == Endpoint.TAIL and edge.endpoint2 == Endpoint.CIRCLE:
#         pag_dict[end[0]].append((start[0], start[1], '--o'))
#     elif edge.endpoint1 == Endpoint.ARROW and edge.endpoint2 == Endpoint.ARROW:
#         pag_dict[end[0]].append((start[0], start[1], '<->'))
#     elif edge.endpoint1 == Endpoint.CIRCLE and edge.endpoint2 == Endpoint.CIRCLE:
#         pag_dict[end[0]].append((start[0], start[1], 'o-o'))
#     else:
#         raise ValueError('LinkType not considered!')

# # Print the PAG dictionary


# for key, value in pag_dict.items():
#     print(f"{key}: {value}")


# Import necessary classes and functions from causal-learn
from causallearn.graph.Dag import Dag
from causallearn.graph.Edge import Edge

from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.DAG2PAG import dag2pag
from causallearn.graph.Endpoint import Endpoint

# Create a DAG with the confounder U
dag = Dag(['A', 'B', 'C', 'U'])

# Add edges (including the confounder U)
dag.add_edge(Edge('U', 'A', Endpoint.TAIL, Endpoint.ARROW))
dag.add_edge(Edge('U', 'B', Endpoint.TAIL, Endpoint.ARROW))
dag.add_edge(Edge('A', 'C', Endpoint.TAIL, Endpoint.ARROW))
dag.add_edge(Edge('B', 'C', Endpoint.TAIL, Endpoint.ARROW))

# Convert the DAG to a PAG
pag = dag2pag(dag, islatent=['U'])

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
