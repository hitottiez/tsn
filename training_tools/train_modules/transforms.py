# additional transforms for okutama-action dataset
import random

from PIL import Image, ImageOps


class GroupRandomVerticalFlip(object):
    """
    Randomly vertical flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in img_group]
            if self.is_flow:
                for i in range(1, len(ret), 2):
                    # invert y_flow pixel values when flipping
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group
