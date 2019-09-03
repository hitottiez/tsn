# PoC2の画像サイズを320x180にリサイズ

import argparse
import glob
import os
import sys
from multiprocessing import Pool

import cv2

from pathlib import Path
from py_broxflow import BroxFlow
from py_tvl1flow import TVL1Flow

tvl1 = TVL1Flow()
brox = BroxFlow()


def extract_tvl1_xy(previous_image, image):
    op_image = tvl1.calc(previous_image, image)
    h, w = op_image.shape[:-1]
    flow_x = op_image[:, :, 0]
    flow_y = op_image[:, :, 1]

    return flow_x, flow_y


def extract_brox(previous_image, image):
    return brox.calc(previous_image, image)


def img_read(path, frame_idx):
    filepath = path.joinpath('{}.jpg'.format(frame_idx))
    return cv2.imread(str(filepath))


def output(image, out_dirpath, filename):
    outfilepath = out_dirpath.joinpath(filename)
    cv2.imwrite(str(outfilepath), image, [cv2.IMWRITE_JPEG_QUALITY, 60])


def processing(movie_frames_dirpath):
    file_count = len(list(movie_frames_dirpath.glob('*.jpg')))
    prev_frame = None
    for index in range(0, file_count):
        print('process {} {}.jpg'.format(movie_frames_dirpath.name, index))
        if index == 0:
            prev_frame = img_read(movie_frames_dirpath, index)
            frame = img_read(movie_frames_dirpath, index + 1)
            tvl1_x, tvl1_y = extract_tvl1_xy(prev_frame, frame)
            # brox = extract_brox(prev_frame, frame)
            frame = prev_frame
        else:
            frame = img_read(movie_frames_dirpath, index)
            tvl1_x, tvl1_y = extract_tvl1_xy(prev_frame, frame)
            # brox = extract_brox(prev_frame, frame)
        output(tvl1_x, movie_frames_dirpath, '{:d}_x.jpg'.format(index))
        output(tvl1_y, movie_frames_dirpath, '{:d}_y.jpg'.format(index))
        # output(brox, movie_frames_dirpath, '{:d}.jpg'.format(index))
        prev_frame = frame
    print('Process End: {}'.format(movie_frames_dirpath.name))


def main():
    # 入力
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirpath', help='/path/to/dataset/')
    parser.add_argument('--worker', type=int, default=4)
    args = parser.parse_args()

    mp_args = []
    p = Path(os.path.join(args.input_dirpath, 'images'))
    mp_args = [movie_frames_dirpath for movie_frames_dirpath in p.glob('*')]
    with Pool(args.worker) as p:
        p.map(processing, mp_args)


if __name__ == '__main__':
    main()
