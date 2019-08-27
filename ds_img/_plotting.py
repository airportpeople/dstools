import matplotlib.pyplot as plt
from ._processing import *


def show_image(img, gray=False):
    '''
    The image needs to be a url, or it needs to be an array. If it's a file, you can use (from skimage)

    image_io = skimage.io.imread(IMAGE_PATH)

    Parameters
    ----------
    img
    gray

    Returns
    -------

    '''
    if not hasattr(img, 'shape'):
        img = url_to_image(img)

    fig, axis = plt.subplots(1)

    if gray:
        axis.imshow(img, cmap='gray')
    else:
        axis.imshow(img)

    plt.grid(None);
    return fig, axis


def multiplot_img(urls, n_cols=3, figsize=(22, 22), titles=None, fontsize=18):
    n_rows = int(np.ceil(len(urls) / n_cols))

    fig = plt.figure(figsize=figsize)

    for i, url in enumerate(urls):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.imshow(url_to_image(url))
        if titles is not None:
            ax.set_title(titles[i], fontsize=fontsize)
        plt.grid(None)

    fig.tight_layout()

    return fig