import json
import urllib
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from glob import glob
from skimage.measure import compare_ssim


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
    if 'https:/' not in url and 'http:/' not in url:
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


def resize(image_path, width, height, background_colors=(255, 255, 255, 255)):
    '''
    Resize PIL image keeping ratio and using white background.
    '''
    image_pil = Image.open(image_path)
    ratio_w = width / image_pil.width
    ratio_h = height / image_pil.height

    if ratio_w < ratio_h:
        # It must be fixed by width
        resize_width = width
        resize_height = round(ratio_w * image_pil.height)

    else:
        # Fixed by height
        resize_width = round(ratio_h * image_pil.width)
        resize_height = height

    image_resize = image_pil.resize((resize_width, resize_height), Image.ANTIALIAS)
    background = Image.new('RGBA', (width, height), background_colors)
    offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
    background.paste(image_resize, offset)

    return background.convert('RGB')


def supervisely_to_df(label_path, agg_only=True):
    '''
    Take a folder of label outputs from Supervisely (supervisely.ly downloads), and codify the information into a dataframe.

    Parameters
    ----------
    label_path : str
        The directory path to the json files

    Returns
    -------
    df_labels : pd.DataFrame
        The output is a Pandas Dataframe containing imaging analysis and scores for use in the near future.

    '''
    filepaths = glob(label_path + "/**")

    image_ids = []
    image_types = []
    image_heights = []
    image_widths = []
    object_titles = []
    object_x1s = []
    object_y1s = []
    object_x2s = []
    object_y2s = []

    for filepath in filepaths:
        with open(filepath) as f:
            label = json.load(f)

        if len(label['objects']) < 1:
            image_ids.append(filepath[filepath.rfind('/') + 1: filepath.rfind('.')])
            image_types.append(label['tags'][0]['name'])
            image_heights.append(label['size']['height'])
            image_widths.append(label['size']['width'])

            object_titles.append(np.nan)
            object_x1s.append(np.nan)
            object_y1s.append(np.nan)
            object_x2s.append(np.nan)
            object_y2s.append(np.nan)

        # Clean up object information, save locations of tags
        for object_ in label['objects']:
            image_ids.append(filepath[filepath.rfind('/') + 1: filepath.rfind('.')])
            image_types.append(label['tags'][0]['name'])
            image_heights.append(label['size']['height'])
            image_widths.append(label['size']['width'])

            object_titles.append(object_['classTitle'])
            object_x1s.append(object_['points']['exterior'][0][0])
            object_y1s.append(object_['points']['exterior'][0][1])
            object_x2s.append(object_['points']['exterior'][1][0])
            object_y2s.append(object_['points']['exterior'][1][1])

    df_labels = pd.DataFrame({'image_id': image_ids,
                              'image_type': image_types,
                              'image_height': image_heights,
                              'image_width': image_widths,
                              'object_title': object_titles,
                              'object_x1': object_x1s,
                              'object_y1': object_y1s,
                              'object_x2': object_x2s,
                              'object_y2': object_y2s, })

    obj_avgs = df_labels.groupby(['image_id', 'object_title']) \
        [['object_x1', 'object_y1', 'object_x2', 'object_y2']] \
        .mean().reset_index()

    # Add columns (a set for each `object_title`) to contain averages for each of the sub-columns
    obj_avgs['object_width'] = (obj_avgs['object_x2'] - obj_avgs['object_x1']).abs()
    obj_avgs['object_height'] = (obj_avgs['object_y2'] - obj_avgs['object_y1']).abs()
    obj_avgs['object_x'] = obj_avgs[['object_x2', 'object_x1']].mean(axis=1)
    obj_avgs['object_y'] = obj_avgs[['object_y2', 'object_y1']].mean(axis=1)

    obj_avgs.index = pd.MultiIndex.from_frame(obj_avgs[['image_id', 'object_title']])
    obj_avgs.drop(columns=['image_id', 'object_title'], inplace=True)

    obj_avgs = obj_avgs.unstack(level=1).reset_index()

    obj_avgs.columns = ['|'.join(col[::-1]) for col in obj_avgs.columns.values]
    obj_avgs.rename(columns={'|image_id': 'image_id'}, inplace=True)
    obj_avgs.columns = [col.replace('object', 'avg') for col in obj_avgs.columns]

    df_labels = df_labels.merge(obj_avgs, on='image_id', how='left')

    if agg_only:
        df_labels = df_labels[[col for col in df_labels.columns
                               if col not in ['object_title', 'object_x1', 'object_y1', 'object_x2', 'object_y2']]] \
            .drop_duplicates().reset_index(drop=True)

    return df_labels