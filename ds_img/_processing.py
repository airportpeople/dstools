import json
import urllib
import cv2
import pandas as pd
import numpy as np
import ast
from PIL import Image
from glob import glob
from skimage.measure import compare_ssim
from ds_util import flatten_list


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
                              'object_y2': object_y2s})

    df_labels['object_width'] = (df_labels['object_x2'] - df_labels['object_x1']).abs()
    df_labels['object_height'] = (df_labels['object_y2'] - df_labels['object_y1']).abs()
    df_labels['object_area'] = df_labels['object_width'] * df_labels['object_height']

    df_labels['object_min_x'] = df_labels[['object_x1', 'object_x2']].min(axis=1)
    df_labels['object_max_x'] = df_labels[['object_x1', 'object_x2']].max(axis=1)
    df_labels['object_min_y'] = df_labels[['object_y1', 'object_y2']].min(axis=1)
    df_labels['object_max_y'] = df_labels[['object_y1', 'object_y2']].max(axis=1)

    obj_avgs = df_labels.groupby(['image_id', 'object_title']).agg({'object_min_x': 'min',
                                                                    'object_max_x': 'max',
                                                                    'object_min_y': 'min',
                                                                    'object_max_y': 'max',
                                                                    'object_width': 'mean',
                                                                    'object_height': 'mean',
                                                                    'object_area': 'mean',
                                                                    'image_type': 'count'}).reset_index()

    # Add columns (a set for each `object_title`) to contain averages for each of the sub-columns
    obj_avgs['object_x_centroid'] = obj_avgs[['object_min_x', 'object_max_x']].mean(axis=1)
    obj_avgs['object_y_centroid'] = obj_avgs[['object_min_y', 'object_max_y']].mean(axis=1)

    obj_avgs.rename(columns={'object_width': 'avg_object_width',
                             'object_height': 'avg_object_height',
                             'object_area': 'avg_object_area',
                             'image_type': 'num_objects'}, inplace=True)

    # Reshape vertical data for objects into horizontal data
    obj_avgs.index = pd.MultiIndex.from_frame(obj_avgs[['image_id', 'object_title']])
    obj_avgs.drop(columns=['image_id', 'object_title'], inplace=True)
    obj_avgs = obj_avgs.unstack(level=1).reset_index()
    obj_avgs.columns = ['|'.join(col[::-1]) for col in obj_avgs.columns.values]
    obj_avgs.rename(columns={'|image_id': 'image_id'}, inplace=True)

    # Merge object data into image dataframe
    df_labels = df_labels.merge(obj_avgs, on='image_id', how='left')
    df_labels['image_area'] = df_labels['image_width'] * df_labels['image_height']

    if agg_only:
        titles = df_labels.groupby('image_id')['object_title'].apply(lambda x: sorted(x.dropna().unique().tolist())) \
                          .reset_index().rename(columns={'object_title': 'object_titles'})
        df_labels = df_labels.merge(titles, on='image_id', how='left')
        df_labels.drop_duplicates(subset=['image_id'], inplace=True)
        df_labels.drop(columns=['object_title', 'object_width', 'object_height', 'object_area',
                                'object_x1', 'object_y1', 'object_x2', 'object_y2',
                                'object_min_x', 'object_max_x', 'object_min_y', 'object_max_y'], inplace=True)

    for kind in ['x_centroid', 'y_centroid']:
        centroids = df_labels[[col for col in df_labels.columns if kind in col]]
        max_centroid = centroids.max(axis=1)
        min_centroid = centroids.min(axis=1)

        middle_centroid = pd.concat((min_centroid, max_centroid), axis=1).mean(axis=1)

        if kind == 'x_centroid':
            loc_kind = 'horizontal'
            half_width = df_labels[f'image_width'] / 2
            df_labels[f'Hero|{loc_kind}_loc_rel'] = (df_labels[f'Hero|object_{kind}'] - middle_centroid) / (middle_centroid - 1)
            df_labels[f'Hero|{loc_kind}_loc_abs'] = (df_labels[f'Hero|object_{kind}'] - half_width) / half_width
        else:
            loc_kind = 'vertical'
            half_height = df_labels[f'image_height'] / 2
            df_labels[f'Hero|{loc_kind}_loc_rel'] = (middle_centroid - df_labels[f'Hero|object_{kind}']) / (middle_centroid - 1)
            df_labels[f'Hero|{loc_kind}_loc_abs'] = (half_height - df_labels[f'Hero|object_{kind}']) / half_height

    for kind in ['x', 'y']:
        border_max = df_labels[[col for col in df_labels.columns if f'object_max_{kind}' in col]].max(axis=1)
        border_min = df_labels[[col for col in df_labels.columns if f'object_min_{kind}' in col]].min(axis=1)

        if kind == 'x':
            white_space = border_min + (df_labels['image_width'] - border_max)
            df_labels['white_space_horizontal'] = white_space / df_labels['image_width']
        else:
            white_space = border_min + (df_labels['image_height'] - border_max)
            df_labels['white_space_vertical'] = white_space / df_labels['image_height']

    df_labels['num_objects'] = df_labels[[col for col in df_labels.columns if 'num_objects' in col]].sum(axis=1)
    df_labels['non-hero_area'] = (df_labels['image_area'] - df_labels['Hero|avg_object_area']) / df_labels['image_area']

    object_titles = df_labels.object_titles.astype(str).unique().tolist()
    object_titles = list(set(flatten_list([ast.literal_eval(x) for x in object_titles])))

    for object_title in object_titles:
        w = df_labels['image_width']
        h = df_labels['image_height']
        x = df_labels[f"{object_title}|object_x_centroid"]
        y = df_labels[f"{object_title}|object_y_centroid"]

        df_labels[f"{object_title}|x_section"] = np.ceil((x / w) * 3)
        df_labels[f"{object_title}|y_section"] = np.ceil((y / h) * 3)

    return df_labels
