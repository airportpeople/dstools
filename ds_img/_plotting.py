import matplotlib.pyplot as plt
from ._processing import *
from matplotlib.patches import Rectangle


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


def plot_rectangle(axis, lower_left_point, width, height, ls='--', lw=3, color='red', fill=False, label=None):
    x = lower_left_point[0]
    y = lower_left_point[1]

    p = Rectangle((x, y), width=width, height=height, ls=ls, lw=lw, color=color, fill=fill, label=label)
    axis.add_patch(p)


def show_imagely(image_id, df_labels, colors=('red', 'orange', 'blue', 'pink', 'yellow')):
    df_ = df_labels[df_labels['object_titles'].apply(len) > 0]
    items = df_[df_.image_id == image_id]['object_titles'].iloc[0]
    colors = list(colors)[:len(items)]

    image_io = skimage.io.imread(f'./images/{image_id}')
    fig, axis = show_image(image_io)
    fig.set_size_inches(12, 12)

    for item, color in zip(items, colors):
        x_centroid = df_labels[df_labels['image_id'] == image_id][f'{item}|object_x_centroid'].iloc[0]
        y_centroid = df_labels[df_labels['image_id'] == image_id][f'{item}|object_y_centroid'].iloc[0]
        width = df_labels[df_labels['image_id'] == image_id][f'{item}|avg_object_width'].iloc[0]
        height = df_labels[df_labels['image_id'] == image_id][f'{item}|avg_object_height'].iloc[0]

        left_x = x_centroid - width / 2
        lower_y = y_centroid - height / 2

        axis.scatter([x_centroid], [y_centroid], color=color, s=50, label=f'{item} Center')
        plot_rectangle(axis, (left_x, lower_y), width, height, color=color, label=f'Average {item}')

    axis.legend()
    print(image_id)

    return fig, axis
