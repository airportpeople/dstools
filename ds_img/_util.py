import urllib
import numpy as np
import cv2
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt


def url_to_image(url, default_url_prefix='https://images-na.ssl-images-amazon.com/images/I/', colors='rgb'):
    '''
    Retrive the image from a url

    Parameters
    ----------
    url
    default_url_prefix

    Returns
    -------


    Source: https://www.pyimagesearch.com/2015/03/02/convert-url-to-image-with-python-and-opencv/
    '''
    # download the image, convert it to a NumPy array, and then read it into OpenCV format
    if 'https:/' not in url:
        url = default_url_prefix + url
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    if colors == 'rgb':
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)[..., ::-1]    # Convert from BGR (OpenCV) to RGB (Matplotlib)
    else:
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image


def equal_images(url1, url2, lenience=0.01, return_diff=False):
    im1 = url_to_image(url1, colors='bgr')
    im2 = url_to_image(url2, colors='bgr')

    if im1.shape != im2.shape:
        return False

    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    score, diff = compare_ssim(im1_gray, im2_gray, full=True)

    if 1 - score <= lenience:
        are_equal = True
    else:
        are_equal = False

    if return_diff:
        return are_equal, diff

    else:
        return are_equal


def show_image(img, gray=False):
    if not hasattr(img, 'shape'):
        img = url_to_image(img)

    if gray:
        axis = plt.imshow(img, cmap='gray')
    else:
        axis = plt.imshow(img)

    plt.grid(None)
    return axis


def multiplot_img(urls, n_cols=3, figsize=(22, 22), titles=None):
    n_rows = int(np.ceil(len(urls) / n_cols))

    fig = plt.figure(figsize=figsize)

    for i, url in enumerate(urls):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.imshow(url_to_image(url))
        if titles is not None:
            ax.set_title(titles[i], fontsize=22)
        plt.grid(None)

    fig.tight_layout()

    return fig