import numpy as np
import pandas as pd


def get_iplot_text(data, names=None, round_digits=2):

    if names is None:
        if hasattr(data, 'columns'):
            names = data.columns.tolist()
            num_cols = [col for col in data.columns if data[col].dropna().shape[0] > 0 and
                        str(data[col].dropna().iloc[0]).replace(".", "").replace("-", '').replace("e", '').isnumeric()]
            data[num_cols] = data[num_cols].round(round_digits)

        else:
            names = [f'var{i + 1}' for i in range(data.shape[1])]
            data = pd.DataFrame(columns=names, data=data)
            num_cols = [col for col in data.columns if data[col].dropna().shape[0] > 0 and
                        str(data[col].dropna().iloc[0]).replace(".", "").replace("-", '').replace("e", '').isnumeric()]
            data[num_cols] = data[num_cols].round(round_digits)

    text_array = np.char.array([names[0]] * data.shape[0])
    text_array = text_array + ": " + np.char.array(data[:, 0], unicode=True) + "<br>"

    for i in range(1, len(names)):
        text_array_ = np.char.array([names[i]] * data.shape[0])
        text_array = text_array + text_array_ + ": " + np.char.array(data[:, i], unicode=True) + "<br>"

    return text_array