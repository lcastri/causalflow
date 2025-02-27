import pickle

import numpy as np
from causalflow.basics.constants import ImageExt, LabelType
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
# DAGDIR = '/home/lcastri/git/causalflow/results/test-obs-21022025_battery/'
# RES = 'res.pkl'
# with open(DAGDIR+RES, 'rb') as f:
#     CM = DAG.load(pickle.load(f))
      
# # CM = CM.filter_alpha(0.00001)
# CM.dag(node_layout = 'dot', node_size = 4, 
#        min_cross_width = 0.5, max_cross_width = 1.5, save_name=DAGDIR+"dag_dot.png")
# CM.ts_dag(node_size = 4, 
#           min_cross_width = 0.5, max_cross_width = 1.5, 
#           x_disp=1.5, y_disp=0.2,
#           save_name=DAGDIR+"ts_dag.png")
# CM.save(DAGDIR + 'res_modified.pkl')



    
# DAGDIR = '/home/lcastri/git/causalflow/results/PD-BC_FINAL/'
# RES = 'res_modified.pkl'
# with open(DAGDIR+RES, 'rb') as f:
#     CM = DAG.load(pickle.load(f))
      
# NODE_COLOR = {'WP': 'lightgray', 
#               'TS': 'lightgray', 
#               'PD': 'orange',  
#               'R_V': 'orange', 
#               '\Delta R_B': 'orange', 
#               'C_S': 'lightgray', 
#               'OBS': 'lightgray'}

# CM.dag(node_layout = 'circular', node_size = 4, 
#        min_cross_width = 0.5, max_cross_width = 1.5, save_name=DAGDIR+"dag_circular", node_color=NODE_COLOR)
# CM.dag(node_layout = 'dot', node_size = 4, 
#        min_cross_width = 0.5, max_cross_width = 1.5, save_name=DAGDIR+"dag_dot", node_color=NODE_COLOR)
# CM.ts_dag(node_size = 4, 
#           min_cross_width = 0.5, max_cross_width = 1.5, 
#           x_disp=1.5, y_disp=0.2,
#           save_name=DAGDIR+"ts_dag", node_color=NODE_COLOR)
# CM.save(DAGDIR + 'res_modified.pkl')

DAGDIR = '/home/lcastri/git/causalflow/results/BC_PRELIMINARY/'
RES = 'res.pkl'
with open(DAGDIR+RES, 'rb') as f:
    CM = DAG.load(pickle.load(f))

NODE_COLOR = {'R_V': 'orange', 
              '\Delta R_B': 'orange',  
              'OBS': 'lightgray'}

# CM.dag(node_layout = 'circular', node_size = 4, 
#        min_cross_width = 0.5, max_cross_width = 1.5, save_name=DAGDIR+"dag_circular", node_color=NODE_COLOR, img_extention=ImageExt.PDF,
#        font_size=15, label_type=LabelType.NoLabels)
node_layout = {'$OBS$': np.array([1.  , 0.95]), 
                       '$R_{V}$': np.array([0.75 , (0.95+0.35)/2]), 
                       '$\\Delta R_{B}$': np.array([ 1., 0.35])}
CM.dag(node_layout = node_layout, node_size = 7, 
       min_cross_width = 1, max_cross_width = 2, save_name=DAGDIR+"dag_dot", node_color=NODE_COLOR, img_extention=ImageExt.PDF,
       font_size=20, label_type=LabelType.Lag)
# CM.ts_dag(node_size = 4, 
#           min_cross_width = 0.5, max_cross_width = 1.5, 
#           x_disp=1.5, y_disp=0.2,
#           save_name=DAGDIR+"ts_dag", node_color=NODE_COLOR, img_extention=ImageExt.PDF)
# CM.save(DAGDIR + 'res_modified.pkl')
