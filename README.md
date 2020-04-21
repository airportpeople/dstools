# dstools

## Initial Setup

1. Install [Anaconda](https://www.anaconda.com/distribution/#macos)
2. Run `conda update --all`
3. Consider the following packages: uszipcode, tensorflow
4. Connect project(s) to GitHub!

### (For Plotly)
1. Install Node with `conda install -c conda-forge nodejs`
2. Install [Plotly](https://plot.ly/python/getting-started/#installation), and do [Jupyterlab extensions](https://plot.ly/python/getting-started/#jupyterlab-support-python-35) (install nodejs first)
3. Install orca with `conda install -c plotly plotly-orca` (for exporting static images of plots)

## Recommendations

- Make sure all code is backed up to GitHub, and all data has at least one redundancy
- Keep environments named by global projects (e.g., company names of clients).
- **Install Snowflake on a separate environment**
- [Always use conda install when possible](https://www.anaconda.com/using-pip-in-a-conda-environment/)
- Don't forget [this](http://scipy-lectures.org/intro/language/reusing_code.html) article! Remember **Put everything in functions in modules, THEN call those functions in a script. KEEP FUNCTIONS AS LOW LEVEL AS POSSIBLE.**
- Remove pyc files from tracking if you need to with `git rm --cached *.pyc` or `find . -name '*.pyc' | xargs -n 1 git rm --cached *.pyc` for recursive. (`*.pyc` should be in the .gitignore anyway.)

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

## Remote Connecting
Windows: `nslookup` to find IP address and computer name, Unix `ifconfig`

## Managing package versioning
Essentially have a working branch and a master branch. Do all your work in the working branch, and then do pull requests from the master branch when you have a working version. Then you can use the normal MAJORUPDATE.MINORUPDATE.PATCH convention.

