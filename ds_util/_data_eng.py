import os
import re
import pandas as pd
import numpy as np
from multiprocessing import Pool, current_process
from datetime import datetime


month_map = {m: datetime(year=2000, month=m, day=1).strftime('%b') for m in range(1, 13)}
table_names = []
num_subtables = []


def df_dump(df, savedir, by_group=None, dfname='df', maxsize=1.5e9, axis=0, pklprotocol=-1, maxrows=1e6, csv_params=None, csv=False,
            overwrite=False):
    '''
    Save a large dataframe as multiple files based on the maximum number of rows, by groups, or the maximum (in buffer) dataframe size.
    
    Parameters
    ----------
    df : pd.DataFrame
        The original data set
    savedir : str
        File path to save the data files
    by_group : str (optional)
        The name of the column with groups by which you'd like to save the data.
    dfname : str (optional)
        A prefix for all the data files to be saved
    maxsize : float (optional)
        The maximum size (in buffer) of each of the data files you'd like to save
    axis : int, in {0, 1} (optional)
        In the end, this is the axis you'll end up concatenating on when you load the data.
    pklprotocol : int, (optional)
        The pickle protocol to pass to `df._to_pickle`
    maxrows : int, (optional)
        The maximum number of rows for each file containing a subset of the data.
    csv_params : dict, (optional)
        The parameters to send to `df.to_csv`
    csv : bool, (optional, default: False)
        Whether you'd like to save as csv files (True) or pickle files (False)
    overwrite : bool, (optional, default: False)
        If you'd like to overwrite data that is already in the directory, 
        or save different files with time stamp as a suffix for each file.
    Returns
    -------
    None
    '''
    file_suffix = '' if overwrite else pd.Timestamp.today().strftime("%Y-%m-%d_%H%M%S")

    def save(df_, filename):
        if csv:
            params = csv_params if csv_params is not None else {}

            if os.path.isfile(filename + '.csv') and not overwrite:
                filename = filename + f'_{file_suffix}.csv'

            params['path_or_buf'] = filename + '.csv'
            df_.to_csv(**params)

        else:
            if os.path.isfile(filename + '.pkl') and not overwrite:
                filename = filename + f'_{file_suffix}.pkl'

            pd.to_pickle(df_, filename + '.pkl', protocol=pklprotocol)

    fullmem = sum(df.memory_usage())

    print('**Total Memory Usage (in bytes): ', fullmem)

    if df.shape[0] > maxrows:
        n_batches = int(np.ceil(df.shape[0] / maxrows))

    elif fullmem < int(maxsize - maxsize * 0.1) or df.shape[0] < maxrows:
        n_batches = 1

    else:
        n_batches = fullmem // int(maxsize - maxsize * 0.1)

    batch_size = df.shape[axis] // n_batches

    if by_group is not None:
        for i, group in enumerate(df[by_group].unique()):
            print(f'({i+1} of {df[by_group].nunique()}) Saving data from {by_group} {group} to {savedir} ...')
            save(df[df[by_group] == group], f'{savedir}/{str(group)}')

        return None

    print(f'Dumping {n_batches} files ...')
    for i in range(n_batches):
        if i == n_batches - 1:
            if axis == 0:
                save(df[i * batch_size:], f'{savedir}/{dfname}_byrows_{i + 1}')
            elif axis == 1:
                save(df.iloc[:, i * batch_size:], f'{savedir}/{dfname}_bycols_{i + 1}')

        else:
            if axis == 0:
                save(df[i * batch_size: (i + 1) * batch_size], f'{savedir}/{dfname}_byrows_{i + 1}')
            elif axis == 1:
                save(df.iloc[:, i * batch_size: (i + 1) * batch_size], f'{savedir}/{dfname}_bycols_{i + 1}')

        print(f'({i + 1} of {n_batches}) Saved data for {dfname}.')

    print('Done!')


def _df_load(work):
    '''
    Load one file ...
    '''

    file, keep_filename, len_prefix, len_suffix, filename_column, savedir, csv_params = work

    print(f"[{current_process().pid}] Loading file {file} from {savedir} ...")

    try:
        df_ = pd.read_pickle(savedir + '/' + file)

    except:
        if file[file.rfind('.') + 1:] in ['gzip', 'bz2', 'zip', 'xz']:
            comp = file[file.rfind('.') + 1:]
        else:
            comp = 'infer'

        params = {'filepath_or_buffer': savedir + '/' + file,
                  'compression': comp}

        params = {**params, **csv_params}

        df_ = pd.read_csv(**params)

    if keep_filename:
        df_[filename_column] = file[len_prefix:len(file) - len_suffix]

    return df_


def df_load(savedir, keep_filename=False, len_prefix=None, len_suffix=4, filename_column='filename', axis=0,
            reset_index=True, csv_params=None, re_pat=".*", n_jobs=1, concat_sort=None):
    '''
    Load multiple files into a Pandas DataFrame (possibly saved using `df_dump`, but not necessarily).
    
    Parameters
    ----------
    savedir : str
        The filepath with the files (only those files) containing the data. Each file should have the *same* columns and the *same* file type. There
        should be no other kinds of file inside the `savedir`.
    keep_filename : bool
        Whether to keep the filename as a column in the data (handy if the files are saved as dates, or something like that).
    len_prefix : int
        If `keep_filename == True`, then for each filename (after the last '/' in the file path) this is the number of unwanted characters at the
        beginning of the file name. If you want all of the filename, you can just leave this as None.
    len_suffix : int
        If `keep_filename == True`, then for each filename (after the last '/' in the file path) this is the number of unwanted characters at the
        end of the file name (including the '.csv' or '.pkl', etc.). If you want all of the filename, you can just leave this as 4.
    filename_column : str
        If `keep_filename == True`, then this is the name of the column where you want to put the filenames.
    axis : int, in {0, 1}
        This is the axis to concatenate the files on to get the final dataset.
    reset_index : bool
        Whether to reset the index in the final data frame after having concatenated.
    csv_params : dict
        The parameters to send to `pd.read_csv`
    re_pat : str
        Regular Expression pattern to match files in the directory
    n_jobs : int
        The number of processes to run

    Returns
    -------
    (pd.DataFrame) The final data set
    '''

    if csv_params is None:
        csv_params = {}

    dirsize = len([f for f in os.listdir(savedir) if f[0] != '.'])

    if dirsize == 0:
        print("There's nothing in here ...")
        return None

    files = sorted([f for f in os.listdir(savedir) if f[0] != '.' and re.match(re_pat, f)])
    if 'bycols' in files[0]:
        axis = 1

    allwork = zip(files,
                  [keep_filename] * len(files),
                  [len_prefix] * len(files),
                  [len_suffix] * len(files),
                  [filename_column] * len(files),
                  [savedir] * len(files),
                  [csv_params] * len(files))

    print(f"Loading {len(files)} files using {n_jobs} jobs...\n")
    if n_jobs == 1:
        dfs = []
        for work in allwork:
            dfs.append(_df_load(work))

    else:
        with Pool(n_jobs) as p:
            dfs = p.map(_df_load, allwork)

    print('Concatenating files into dataframe ...')
    df = pd.concat(dfs, axis=axis, sort=concat_sort)

    if reset_index:
        df.reset_index(inplace=True, drop=True)

    return df


def df_join_array(df, array, column_names):
    '''
    Join a Numpy array with a Pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    array : np.array
    column_names : list
        A list of the column names to use on the Numpy array

    Returns
    -------
    (pd.DataFrame) The final data frame.
    '''
    df = pd.concat([df, pd.DataFrame(columns=column_names)], sort=True)
    df.loc[:, column_names] = array

    return df


def define_bins(values, bins=(0, 2, 5, 10, 50, 100, 500, 2000)):
    '''
    ...
    **NOTE: An older version of this method exists in DataScience.sales_ensemble.scriptlib.util. UPDATE THE OLDER IF NECESSARY.**
    ...
    You can use something like
        sales.ensembledata['SalesCountsSection'] = define_category(sales.ensembledata.SalesActualCount30.values)
        sales.plot_report_by_col('SalesCountsSection', ['SalesEstimate30', 'Class2Reg_Poisson', 'Class2Reg_LM'],
                                 ['MAAPE', 'sMAPE'], 'count', 'index')
    :param values:
    :param bins:
    :return:
    '''
    bins = np.array(bins)
    bins_values = np.array([bins] * len(values))
    values_bins = np.array([values] * len(bins)).T
    mask = values_bins >= bins_values
    bin_index = np.sum(mask, axis=1) - 1
    bins_defined = bins[bin_index]

    return bins_defined


def clean_json(raw_, prev_key='', idx_separator='__'):
    '''
    Clean up a raw json so that json2table outputs a palatable set of tables for Pandas to turn to dataframes.

    Parameters
    ----------
    raw_ : dict or json
        The raw json you want to have cleaned up.

    prev_key : str, DON'T CHANGE
        This is for the recursion of the function.

    idx_separator : str, optional
        In recursion, the function will separate dictionary/json keys. Pick a value that will not show up in any of these strings.

    Returns
    -------
    clean : dict
        A clean json!

    '''
    global table_names
    global num_subtables

    for k in raw_.keys():
        if isinstance(raw_[k], dict):
            # Instead of reading a dictionary as it's own values, read it as a sub-json (in a list)
            # This allows json2table to convert it correctly into its own sub-table
            raw_[k] = [clean_json(raw_[k], prev_key=k, idx_separator=idx_separator)]
            table_names.append(prev_key + k)
            num_subtables.append(1)

        elif isinstance(raw_[k], list) and len(raw_[k]) > 0 and not isinstance(raw_[k][0], dict):
            # If the value is a list of non-dictionaries, json2table wants to turn to a string.
            # Take the list out from sublevel, and add new keys to the 'super' dictionary pointing to the values by index
            sub2super = {}
            for i in range(len(raw_[k])):
                sub2super[str(k) + idx_separator + str(i)] = raw_[k][i]

            # Sometimes, this will be part of a sub-dictionary, otherwise, it's the main raw_ json
            try:
                raw_[prev_key] = {**raw_[prev_key], **sub2super}
            except KeyError:
                raw_ = {**raw_, **sub2super}

            # Once we pull the values out of key `k`, we don't need that anymore
            _ = raw_.pop(k)

        elif isinstance(raw_[k], list) and len(raw_[k]) > 0 and isinstance(raw_[k][0], dict):
            # If the value is a list of dictionaries, don't do anything (because this is the format json2table likes)
            # and add table names and counts
            table_names.append(prev_key + k)
            num_subtables.append(len(raw_[k]))

    return raw_


def flatten_multilevel_cols(df, separator="_", inplace=False):
    '''
    Get a single level of columns from pandas MultiIndex columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame with a multiindex to be droped to a single level
    separator : str, (optional, default: '|')
        The preferred separator between levels for each consolidated column name (e.g., col_level1|col_level2)
    inplace : bool, (optional, default: False)
        Whether to run the script on the data frame's columns in place

    Returns
    -------
    If `inplace == True` : (None) This works on the columns in the dataframe
    If `inplace == False` : (list-like) The list of new column names
    '''

    new_columns = [separator.join(col) for col in df.columns.values]
    if inplace:
        df.columns = new_columns
        return None

    return new_columns


def get_intact_columns(df, max_perc_missing=0.25):
    '''
    Given a dataframe, which columns have less than a specified amount of missing values?

    Parameters
    ----------
    df : pandas.DataFrame
        Data

    max_perc_missing: float
        Maximum acceptable proportion of NULL values in the column

    Returns
    -------
    (list) A list of column names abiding by rule given
    '''

    column_perc_missing = df.isna().sum() / df.shape[0]

    return column_perc_missing[column_perc_missing <= max_perc_missing].index.tolist()


def get_interval_bins(series, bins, string_labels=True):
    '''

    Parameters
    ----------
    series
    bins : int, list
        If int, use qcut for approximately equally sized bins (i.e., about the same number of items in each bin)
        If list, do not include the minimum or maximum. These are the "cuts" inbetween. So,
            bins = [5, 10, 15]  -->  [a.min(), 5), [5, 10), [10, 15), [15, a.max() + 1)
    string_labels : bool

    Returns
    -------

    '''
    if isinstance(bins, int):
        a = pd.qcut(series, bins)

        if string_labels:
            a = a.astype(str).str.strip('()[]').str.replace(',', ' -')

        return a

    elif isinstance(bins, list):
        bins = [b for b in bins if series.min() < b < series.max()]
        bins = [series.min()] + bins + [series.max() + 1]

        if string_labels:
            labels = [str(bins[i]) + ' - ' + str(bins[i + 1]) for i in range(len(bins) - 1)]
        else:
            labels = None

        return pd.cut(series, bins=bins, labels=labels, right=False)

    else:
        raise AttributeError("`bins` must be an integer or array.")
