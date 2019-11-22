# dstools

## Initial Setup

1. Install [Anaconda](https://www.anaconda.com/distribution/#macos)
2. Run `conda update --all`
3. Install Node with `conda install -c conda-forge nodejs`
4. Install [Plotly](https://plot.ly/python/getting-started/#installation), and do [Jupyterlab extensions](https://plot.ly/python/getting-started/#jupyterlab-support-python-35) (install nodejs first with)
5. Consider the following packages: uszipcode, tensorflow, 

## Recommendations

- Keep environments named by global projects (e.g., company names of clients).
- **Only install Snowflake on a separate environment**
- [Use conda first then pip](https://www.anaconda.com/using-pip-in-a-conda-environment/)

## Set up new (conda) environment

1. Use the syntax `conda create -n <myenv> python=3.7 scipy=0.15.0 astroid babel`
2. Activate the environment `conda activate <myenv>`
3. Run `conda install ipykernel`
4. Then `ipython kernel install --user --name=<myenv>`
5. Then `conda deactivate` and it should be in there

## IPython Kernels

To see all kernels, `jupyter kernelspec list`.
To remove a kernel `jupyter kernelspec remove <myenv>`.

