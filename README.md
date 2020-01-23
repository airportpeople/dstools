# dstools

## Initial Setup

1. Install [Anaconda](https://www.anaconda.com/distribution/#macos)
2. Run `conda update --all`
3. Install Node with `conda install -c conda-forge nodejs`
4. Install [Plotly](https://plot.ly/python/getting-started/#installation), and do [Jupyterlab extensions](https://plot.ly/python/getting-started/#jupyterlab-support-python-35) (install nodejs first)
5. Install orca with `conda install -c plotly plotly-orca` (for exporting static images of plots)
6. Consider the following packages: uszipcode, tensorflow
7. Connect project(s) to GitHub!

## Recommendations

- Make sure all code is backed up to GitHub, and all data has at least one redundancy
- Keep environments named by global projects (e.g., company names of clients).
- **Install Snowflake on a separate environment**
- [Always use conda install when possible](https://www.anaconda.com/using-pip-in-a-conda-environment/)

## Set up new (conda) environment

1. First, connect your project to a GitHub repo for VCS and code backup
2. Use the syntax `conda create -n <myenv> python=3.7 scipy=0.15.0 astroid babel`
3. Activate the environment `conda activate <myenv>`
4. Run `conda install ipykernel`
5. Then `ipython kernel install --user --name=<myenv>`
6. Then `conda deactivate` and it should be in there
7. (Optional) **If you're installing Snowflake, install it now!** Then look at installing [Snowflake SQL Alchemy](https://docs.snowflake.net/manuals/user-guide/sqlalchemy.html).

## IPython Kernels

To see all kernels, `jupyter kernelspec list`.
To remove a kernel `jupyter kernelspec remove <myenv>`.

