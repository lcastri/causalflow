from rpy2.robjects import r, pandas2ri
from rpy2.robjects.vectors import StrVector, IntVector
from rpy2.robjects.packages import importr
import pandas as pd

# Example DAG in Python dictionary format
DAG = {
    'X_1': [('X_1', -1, '-->'), ('X_2', -1, '-->'), ('X_3', 0, '-->')],
    'X_2': [],
    'X_3': [('X_4', -1, '-->'), ('X_2', -1, '-->')],
    'X_4': [('X_4', -1, '-->')]
}

# Convert the DAG into a format that can be understood by rpy2
dag_df = pd.DataFrame(columns=['From', 'To', 'Type'])

for node, connections in DAG.items():
    for conn in connections:
        from_var = node
        to_var, lag, arrow = conn
        dag_df = dag_df.append({'From': from_var, 'To': to_var, 'Type': arrow}, ignore_index=True)

# Convert pandas DataFrame to R data.frame
pandas2ri.activate()
r_dataframe = pandas2ri.py2rpy_pandasdataframe(dag_df)

# Print the R dataframe (optional)
print(r_dataframe)



# Import pcalg package
pcalg = importr('pcalg')

# Convert DAG to PAG using pcalg.dag2pag
pag_result = pcalg.dag2pag(r_dataframe)

# Convert pag_result back to Python (if needed)
pag_df = pandas2ri.ri2py_dataframe(pag_result)

# Print the resulting PAG
print(pag_df)
