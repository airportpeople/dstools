import sys
from glob import glob

sys.path.append('/Users/jtw/GitStuff')

for p in [x for x in glob('/Users/jtw/GitStuff/**') if '.' not in x]:
    sys.path.append(p)

from multiprocessing import Pool, current_process
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
%matplotlib inline

# Barchart: `fig = go.Figure(data=go.Bar(y=[2, 3, 1]))`
# Scatter Plot: `fig = px.scatter(df, x="var1", y="var2")`
# fig.show()

# Run in different cell
sns.set_context('notebook')
sns.set_style('darkgrid')
sns.set(rc={'patch.edgecolor': 'w', 'patch.force_edgecolor': True, 'patch.linewidth': 1})
plt.rcParams['figure.figsize'] = (15, 8)

# New project? use `conda create -n myenv python=3.7 scipy=0.15.0 astroid babel`
# Go in, run `conda install ipykernel`
# Then `ipython kernel install --user --name=<any_name_for_kernel>`
# Then `conda deactivate` and it should be in there