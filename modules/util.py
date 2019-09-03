#-*- coding: utf-8 -*-

# API共通関数
# ==============================================================================

import numpy as np

XMIN = 0
YMIN = 1
XMAX = 2
YMAX = 3


def square_global_roi(xmin, ymin, xmax, ymax):
    box_width = xmax - xmin
    box_height = ymax - ymin
    offset = max(box_width, box_height)
    new_xmin = xmin - offset
    new_ymin = ymin - offset
    new_xmax = xmax + offset
    new_ymax = ymax + offset
    return square_roi(new_xmin, new_ymin, new_xmax, new_ymax)


def square_roi(xmin, ymin, xmax, ymax):
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2

    crop_w = xmax - xmin
    crop_h = ymax - ymin

    crop_size = crop_w
    if crop_h > crop_w:
        crop_size = crop_h

    half_crop_size = crop_size // 2
    xmin = center_x - half_crop_size
    xmax = center_x + half_crop_size
    ymin = center_y - half_crop_size
    ymax = center_y + half_crop_size
    return (xmin, ymin, xmax, ymax)


def extract_flow_local_roi(flow_x, flow_y, box):
    """
    create local roi cropped flow image (for numpy image)
    image:
        numpy array image
    box:
        list of [xmin, ymin, xmax, ymax]
    """
    flow_x_roi = extract_local_roi(flow_x, box)
    flow_y_roi = extract_local_roi(flow_y, box)
    if flow_x_roi is None or flow_y_roi is None:
        return None
    else:
        return (flow_x_roi, flow_y_roi)


def extract_flow_global_roi(flow_x, flow_y, box):
    """
    create global roi cropped flow image (for numpy image)
    image:
        numpy array image
    box:
        list of [xmin, ymin, xmax, ymax]
    """
    flow_x_roi = extract_global_roi(flow_x, box)
    flow_y_roi = extract_global_roi(flow_y, box)
    if flow_x_roi is None or flow_y_roi is None:
        return None
    else:
        return (flow_x_roi, flow_y_roi)


def extract_local_roi(image, box):
    """
    create local roi cropped image (for numpy image)
    image:
        numpy array image
    box:
        list of [xmin, ymin, xmax, ymax]
    """
    xmin, ymin, xmax, ymax = square_roi(
        box[XMIN], box[YMIN], box[XMAX], box[YMAX])

    # fill out outside bbox with specified color
    # img = np.copy(image)
    # img[ymin:ymax,      xmin:box[XMIN]] = outside_bbox_color
    # img[ymin:ymax,      box[XMAX]:xmax] = outside_bbox_color
    # img[ymin:box[YMIN], xmin:xmax] = outside_bbox_color
    # img[box[YMAX]:ymax, xmin:xmax] = outside_bbox_color

    return _extract_roi(image, xmin, ymin, xmax, ymax)


def extract_global_roi(image, box):
    """
    create global roi cropped image (for numpy image)
    image:
        numpy array image
    box:
        list of [xmin, ymin, xmax, ymax]
    """
    xmin, ymin, xmax, ymax = square_global_roi(
        box[XMIN], box[YMIN], box[XMAX], box[YMAX])

    return _extract_roi(image, xmin, ymin, xmax, ymax)


def _extract_roi(image, xmin, ymin, xmax, ymax):
    height, width = image.shape[:2]
    new_xmin, new_ymin, new_xmax, new_ymax = \
        xmin, ymin, xmax, ymax
    if xmin < 0:
        new_xmin = 0
    if xmax >= width:
        new_xmax = width - 1
    if ymin < 0:
        new_ymin = 0
    if ymax >= height:
        new_ymax = height - 1

    roi = image[new_ymin:new_ymax, new_xmin:new_xmax]

    if xmin < 0:
        for _ in range(abs(xmin)):
            roi = np.insert(roi, 0, 0, axis=1)
    if xmax >= width:
        insert_point = len(roi[0])
        for _ in range(abs(xmax - width)):
            roi = np.insert(roi, insert_point, 0, axis=1)
    if ymin < 0:
        for _ in range(abs(ymin)):
            roi = np.insert(roi, 0, 0, axis=0)
    if ymax >= height:
        insert_point = len(roi)
        for _ in range(abs(ymax - height)):
            roi = np.insert(roi, insert_point, 0, axis=0)

    roi_h, roi_w = roi.shape[:2]
    if roi_h < 1 or roi_w < 1:
        raise Exception('Incorrect ROI image')
    return roi


def extract_predict(predict_results, roi_images, before_fc):
    """
    top_kまでの識別結果を抽出し、レスポンスデータを作成する
    """
    results = []
    count = 0
    for roi_image in roi_images:
        if roi_image is None:
            results.append({
                'success': False,
                'scores': [],
                'before_fc_features': [],
            })
        else:
            results.append({
                'success': True,
                'scores': predict_results[count].tolist(),
                'before_fc_features': before_fc[count].tolist()
            })
            count += 1
    return results
