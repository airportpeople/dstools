import os
import pandas as pd
import numpy as np


def df_dump(df, savedir, by_group=None, dfname='df', maxsize=1.5e9, axis=0, pklprotocol=-1, maxrows=np.inf, csv_params=None, csv=False):

    def save(df_, filename):
        if csv:
            params = csv_params if csv_params is not None else {}
            params['path_or_buf'] = filename + '.csv'
            df_.to_csv(**params)

        else:
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

    if len([f for f in os.listdir(savedir) if f[0] != '.']) > 0:
        print('There are already items saved here ... delete them, or move them, and then run this again.')
        return None

    if by_group is not None:
        for i, group in enumerate(df[by_group].unique()):
            print(f'({i+1} of {df[by_group].nunique()}) Saving data from {by_group} {group} to {savedir} ...')
            save(df[df[by_group] == group], f'{savedir}/{str(group)}')

        return None

    print('Dumping files ...')
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


def df_load(savedir, keep_filename=False, len_prefix=None, len_suffix=4, featurecol_name='filename', axis=0,
            reset_index=True, csv_params=None):

    if csv_params is None:
        csv_params = {}

    dirsize = len([f for f in os.listdir(savedir) if f[0] != '.'])

    if dirsize == 0:
        print("There's nothing in here ...")
        return None

    files = sorted([f for f in os.listdir(savedir) if f[0] != '.'])
    if 'bycols' in files[0]:
        axis = 1

    try:
        df = pd.read_pickle(savedir + '/' + files[0])

    except:
        if files[0][files[0].rfind('.')+1:] in ['gzip', 'bz2', 'zip', 'xz']:
            comp = files[0][files[0].rfind('.')+1:]
        else:
            comp = 'infer'

        params = {'filepath_or_buffer': savedir + '/' + files[0],
                  'compression': comp}

        params = {**params, **csv_params}

        df = pd.read_csv(**params)

    if keep_filename:
        df[featurecol_name] = files[0][len_prefix:len(files[0])-len_suffix]

    print(f'Loaded file 1 of {dirsize}.')

    for i, file in enumerate(files[1:]):
        try:
            df_ = pd.read_pickle(savedir + '/' + file)

        except:
            if file[file.rfind('.')+1:] in ['gzip', 'bz2', 'zip', 'xz']:
                comp = file[file.rfind('.')+1:]
            else:
                comp = 'infer'

            params = {'filepath_or_buffer': savedir + '/' + file,
                      'compression': comp}

            params = {**params, **csv_params}

            df_ = pd.read_csv(**params)

        if keep_filename:
            df_[featurecol_name] = file[len_prefix:len(file)-len_suffix]

        df = pd.concat((df, df_), axis=axis, sort=True)

        print(f'Loaded file {i + 2} of {dirsize}.')

    if reset_index:
        df.reset_index(inplace=True, drop=True)

    return df


def df_join_array(df, array, column_names):

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