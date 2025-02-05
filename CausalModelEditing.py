import pickle
from utils import *
from causalflow.graph.DAG import DAG

# node_classification = {
#     list(NODES).index(NODES.TOD): "space_context",
#     list(NODES).index(NODES.RV): "system",
#     list(NODES).index(NODES.RB): "system",
#     list(NODES).index(NODES.CS): "space_context",
#     list(NODES).index(NODES.PD): "system",
#     list(NODES).index(NODES.ELT): "system",
#     list(NODES).index(NODES.OBS): "space_context",
#     list(NODES).index(NODES.WP): "space_context",
# }


# NODE_COLOR = {}
# for node, classification in node_classification.items():
#     if classification == "system":
#         NODE_COLOR[list(NODES)[node].value] = 'orange'
#     elif classification == "space_context":
#         NODE_COLOR[list(NODES)[node].value] = 'lightgray'

# DAGDIR = '/home/lcastri/git/causalflow/results/BL100_21102024_new/'
# RES = 'res_original.pkl'
DAGDIR = '/home/lcastri/git/causalflow/results/5vars_test/'
RES = 'res.pkl'
with open(DAGDIR+RES, 'rb') as f:
    CM = DAG.load(pickle.load(f))
      
# CM = CM.filter_alpha(0.00001)
CM.dag(node_layout = 'dot', node_size = 4, 
       min_cross_width = 0.5, max_cross_width = 1.5, save_name=DAGDIR+"dag_dot.png")
CM.ts_dag(node_size = 4, 
          min_cross_width = 0.5, max_cross_width = 1.5, 
          x_disp=1.5, y_disp=0.2,
          save_name=DAGDIR+"ts_dag.png")
CM.save(DAGDIR + 'res_modified.pkl')
