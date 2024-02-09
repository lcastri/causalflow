import glob
import os
from pathlib import Path
from subprocess import Popen, PIPE
import pandas as pd
import networkx as nx

DSCORE = 0.3

def clean(dir_path):    
    files = glob.glob(dir_path + '/args/*')
    for f in files: os.remove(f)
    os.removedirs(dir_path+"/args")
    
    files = glob.glob(dir_path + '/results/*')
    for f in files: os.remove(f)
    os.removedirs(dir_path+"/results")
    

def runTiMINo(data: pd.DataFrame, max_lag: int, alpha: float):
    # Remove all arguments from directory
    dir_path = os.path.dirname(os.path.realpath(__file__))
    Path(dir_path + "/args").mkdir(exist_ok=True)
    Path(dir_path + "/results").mkdir(exist_ok=True)
        
    script = dir_path + "/timino.R"
    r_arg_list = []
        
    # COMMAND WITH ARGUMENTS
    data.to_csv(dir_path + "/args/data.csv", index=False)
    r_arg_list.append(dir_path + "/args/data.csv")
    r_arg_list.append(str(alpha))
    r_arg_list.append(str(max_lag))

    r_arg_list.append(dir_path)
    cmd = ["Rscript", script] + r_arg_list

    p = Popen(cmd, cwd="./", stdin=PIPE, stdout=PIPE, stderr=PIPE)
        
    # Return R output or error
    output, error = p.communicate()
    print(output.decode('utf-8'))
    if p.returncode == 0:
        g_df = pd.read_csv(dir_path + "/results/result.csv", header=0, index_col=0)
        print(g_df)
        clean(dir_path)
        return g_df

    else:
        clean(dir_path)
        print('R Error:\n {0}'.format(error.decode('utf-8')))
        exit(0)
        
def dataframe_to_graph(nodes, df):
    ghat = nx.DiGraph()
    ghat.add_nodes_from(nodes)
    for name_x in df.columns:
        if df[name_x].loc[name_x] > 0:
            ghat.add_edges_from([(name_x, name_x)])
        for name_y in df.columns:
            if name_x != name_y:
                if df[name_y].loc[name_x] == 2:
                    ghat.add_edges_from([(name_x, name_y)])
    return ghat