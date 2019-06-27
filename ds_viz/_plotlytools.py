import numpy as np


def get_iplot_text(data, names=None):
    if names is None:
        if hasattr(data, 'columns'):
            names = data.columns.tolist()
            data = data.values
        else:
            data = np.array(data)
            names = [f'var{i + 1}' for i in range(data.shape[1])]

    text_array = np.char.array([names[0]] * data.shape[0])
    text_array = text_array + ": " + np.char.array(data[:, 0], unicode=True) + "<br>"

    for i in range(1, len(names)):
        text_array_ = np.char.array([names[i]] * data.shape[0])
        text_array = text_array + text_array_ + ": " + np.char.array(data[:, i], unicode=True) + "<br>"

    return text_array