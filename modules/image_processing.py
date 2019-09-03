# coding:utf-8

import cv2
import numpy as np

def decode_image(image_file):
    """
    画像をデコードする
    """
    nparr = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        print('Invalid Request Parameter[Decode `image` failed.]')
    return image
