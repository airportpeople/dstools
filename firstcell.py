from pathlib import Path
home = str(Path.home())

import sys
import gc
from glob import glob

sys.path.append(home + '/GitStuff')

for p in [x for x in glob(home + '/GitStuff/**') if '.' not in x]:
    sys.path.append(p)

from multiprocessing import Pool, current_process
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
# %matplotlib inline

# Barchart: `fig = go.Figure(data=go.Bar(y=[2, 3, 1]))`
# Scatter Plot: `fig = px.scatter(df, x="var1", y="var2")`
# fig.show()

# Run in different cell
sns.set_context('notebook')
sns.set_style('darkgrid')
sns.set(rc={'patch.edgecolor': 'w', 'patch.force_edgecolor': True, 'patch.linewidth': 1})
plt.rcParams['figure.figsize'] = (15, 8)