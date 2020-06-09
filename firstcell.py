# Run %load ~/PATH/TO/firstcell.py
from pathlib import Path
home = str(Path.home())

import sys
import gc
import json
from glob import glob

sys.path.append(home + '/GitStuff')

for p in [x for x in glob(home + '/GitStuff/**') if '.' not in x]:
    sys.path.append(p)

from multiprocessing import Pool, current_process
from importlib import reload
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
# %matplotlib inline

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Barchart: `fig = go.Figure(data=go.Bar(y=[2, 3, 1]))`
# Scatter Plot: `fig = px.scatter(df, x="var1", y="var2")`
# fig.show()

sns.set_context('notebook')
sns.set_style('darkgrid')
sns.set(rc={'patch.edgecolor': 'w',
            'patch.force_edgecolor': True,
            'patch.linewidth': 1,
            'axes.grid': True,
            'axes.grid.axis': 'both',
            'axes.spines.left': False})
plt.rcParams['figure.figsize'] = (15, 8)

# from IPython.display import display

# with pd.option_context('display.max_rows', 300, 'display.max_columns', 100):
#     display(df.iloc[:300])