import os
import os.path
import numpy as np
from pathlib import PurePath

from PIL import Image

import pandas as pd
import torch
import torch.utils.data as data
from modules.util import square_global_roi, square_roi

import random

XMIN = 0
YMIN = 1
XMAX = 2
YMAX = 3


class ActionRecord(object):
    """
    動画情報のtxtファイルに書かれた情報を管理する
    txtの中身はスペース区切りの以下のフォーマット   
    <lavel_filepath> <frame_num> <label1> <label2> <label3> <label4>
    """

    def __init__(self, row):
        self._data = row
        p = PurePath(row[0])

        df = pd.read_csv(str(p), header=None, sep=' ')
        df.columns = ('local_id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame')
        self._df = df
        self.start_frame = df.frame.min()

        # /path/to/1.1.1/0.txt -> 1.1.1
        movie_name = p.parent.name

        # /path/to/dataset/tsn-labels/train/1.1.1/0.txt
        # to
        # /path/to/dataset/images/1.1.1/
        p = p.parent.parent.parent.parent.joinpath(
                'images/' + movie_name)

        self.imgpath = str(p)

    def person_roi_range(self, frame_index):
        # 1列だけデータ取得
        series = self._df[self._df.frame == frame_index].iloc[0]

        bbox = (series.xmin, series.ymin, series.xmax, series.ymax)
        square = square_roi(bbox[XMIN], bbox[YMIN], bbox[XMAX], bbox[YMAX])
        return bbox, square

    @property
    def end_frame(self):
        return self.num_frames + self.start_frame

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label1(self):
        return int(self._data[2])

    @property
    def label2(self):
        return int(self._data[3])

    @property
    def label3(self):
        return int(self._data[4])

    @property
    def label4(self):
        return int(self._data[5])

class TSNMultiDataSet(data.Dataset):

    def __init__(self, rgb_dataset, flow_dataset):
        self.rgb_dataset = rgb_dataset
        self.flow_dataset = flow_dataset

    def __getitem__(self, index):
        rgb_data = self.rgb_dataset.__getitem__(index)
        flow_data = self.flow_dataset.__getitem__(index)
        return rgb_data, flow_data

    def __len__(self):
        return self.rgb_dataset.__len__()


class VideoFrame(object):
    """
    データローダーが__getitem__関数を呼ぶ際のindexに紐づく、
    videoのインデックスとその開始フレームのインデックスを管理するクラス
    """
    def __init__(self, record_index, frame_index):
        self.record_index = record_index
        self.frame_index = frame_index


class TSNDataSetCustom(data.Dataset):
    """
    オリジナルとはデータロードの方法が異なる
    オリジナル
        ：1処理（1動画）あたりnum_segments x new_lengthのフレームを処理
    カスタマイズ
            ：動画を先頭stride_step_num飛ばしにnum_segmentsだけロードする。次のデータは、slide_numだけ後ろにスライドした
            stride_step_num飛ばしのnum_segments分のフレームを1データとして扱う。
              1動画あたり、
              <1 + (frame_num - (stride_step_num * (num_segments - 1) + num_segments)) // slide_num>
              のデータがロードされる。
        new_length: 削除 (1固定)
        slide_num: 次のデータロードするときの、前回読み込んだフレーム範囲からのスライド数
        num_segments: 1処理あたりに返却するframeの数
        stride_step_num: num_segments個のフレームを取るときに、何個飛ばしにフレームを取るかの値
    """

    def __init__(self, list_file,
                 num_segments=3, slide_num=1, modality='RGB',
                 image_tmpl='{:d}.jpg', transform=None,
                 stride_step_num=0,
                 does_use_global_img=False,
                 train_mode=True):
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = 1
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.slide_num = slide_num
        self.stride_step_num = stride_step_num
        self.does_use_global_img = does_use_global_img
        self.train_mode = train_mode
        self.crop_rate = 0.8

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff
        self._parse_list()

    def _load_image(self, directory, idx):
        """
        same as original
        """
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            path = os.path.join(directory, self.image_tmpl.format(idx))
            return [Image.open(path).convert('RGB')]
        elif self.modality == 'Flow':
            x_path = os.path.join(directory, self.image_tmpl.format(idx, 'x'))
            y_path = os.path.join(directory, self.image_tmpl.format(idx, 'y'))
            return [Image.open(path).convert('L') for path in [x_path, y_path]]

    def _parse_list(self):
        """
        same as original
        """
        self.action_records = [ActionRecord(x.strip().split(' '))
                               for x in open(self.list_file)]
        self._calc_data_len()

    def _random_crop_offset_rate(self):
        w_rate = random.randint(0, 5) / 5
        h_rate = random.randint(0, 5) / 5

        def plus_or_minus():
            return 1 if random.random() < 0.5 else - 1

        return w_rate * plus_or_minus(), h_rate * plus_or_minus()

    def _random_crop_area(self, bbox, square_roi, w_offset_rate,
                          h_offset_rate, crop_rate):
        scale_w = (square_roi[XMAX] - square_roi[XMIN]) * (1 - crop_rate)
        scale_h = (square_roi[YMAX] - square_roi[YMIN]) * (1 - crop_rate)

        w_offset = int(scale_w * w_offset_rate)
        h_offset = int(scale_h * h_offset_rate)

        return (square_roi[XMIN] + w_offset, square_roi[YMIN] + h_offset,
                square_roi[XMAX] + w_offset, square_roi[YMAX] + h_offset)

    def _crop_global_image(self, imgs, bbox, w_offset_rate, h_offset_rate):
        roi = square_global_roi(bbox[XMIN], bbox[YMIN],
                                bbox[XMAX], bbox[YMAX])

        if self.train_mode:
            roi = self._random_crop_area(bbox, roi, w_offset_rate,
                                         h_offset_rate, self.crop_rate)
        ret_imgs = []
        for img in imgs:
            ret_imgs.append(img.crop(roi))
        return ret_imgs

    def _crop_person_image(self, imgs, bbox, square_roi, w_offset_rate,
                           h_offset_rate):
        if self.train_mode:
            square_roi = self._random_crop_area(bbox, square_roi,
                                                w_offset_rate, h_offset_rate,
                                                self.crop_rate)
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            img = imgs[0]
            cropped_img = img.crop(square_roi)
            return [cropped_img]
        elif self.modality == 'Flow':
            x_img, y_img = imgs
            x_cropped_img = x_img.crop(square_roi)
            y_cropped_img = y_img.crop(square_roi)
            return [x_cropped_img, y_cropped_img]

    def _paint_outside_bbox(self, cropped_img, bbox, square_roi, color=0):
        # reset position from original image to cropped image
        bbox_xmin = bbox[XMIN] - square_roi[XMIN]
        bbox_ymin = bbox[YMIN] - square_roi[YMIN]
        bbox_xmax = cropped_img.width - bbox_xmin
        bbox_ymax = cropped_img.height - bbox_ymin

        left_right_outside_pixel = Image.new(
            'RGB',
            size=(bbox_xmin, cropped_img.height),
            color=(color, color, color))
        top_bottom_outside_pixel = Image.new(
            'RGB',
            size=(cropped_img.width, bbox_ymin),
            color=(color, color, color))

        cropped_img.paste(top_bottom_outside_pixel, (0, 0))
        cropped_img.paste(top_bottom_outside_pixel, (0, bbox_ymax))
        cropped_img.paste(left_right_outside_pixel, (0, 0))
        cropped_img.paste(left_right_outside_pixel, (bbox_xmax, 0))

    def _calc_data_len(self):
        """
        1動画あたり取得するデータは
        1 + (frame_num - <1データあたりの範囲>) // slide_num
        1 + (frame_num - (stride_step_num * (num_segments - 1) + num_segments)) // slide_num
        """
        self._data_len = 0
        self._frames = []
        datum_len = self.stride_step_num * \
            (self.num_segments - 1) + self.num_segments
        for record_index, action_record in enumerate(self.action_records):
            start_len = self._data_len
            self._data_len += 1 + (action_record.num_frames -
                                   datum_len) // self.slide_num
            for i, _ in enumerate(range(start_len, self._data_len)):
                '''
                    action_record:
                        local_id  xmin  ymin  xmax  ymax  frame
                                4  2184   182  2243   275     70
                                4  2184   182  2245   285     71
                                ...
                    then:
                        start_frame = 70
                '''
                frame_index = i * self.slide_num + action_record.start_frame
                self._frames.append(VideoFrame(record_index=record_index,
                                               frame_index=frame_index))

    def __getitem__(self, index):
        frame = self._frames[index]
        record = self.action_records[frame.record_index]
        frame_index = int(frame.frame_index)

        global_images = []
        local_images = []
        w_rate, h_rate = self._random_crop_offset_rate()
        for _ in range(self.num_segments):
            seg_imgs = self._load_image(record.imgpath, frame_index)
            # person roi range
            bbox, square_roi = record.person_roi_range(frame_index)
            local_imgs = self._crop_person_image(seg_imgs, bbox, square_roi,
                                                 w_rate, h_rate)
            local_images.extend(local_imgs)

            if self.does_use_global_img:
                global_imgs = self._crop_global_image(
                    seg_imgs, bbox, w_rate, h_rate)
                global_images.extend(global_imgs)

            if frame_index < record.end_frame:
                frame_index += self.stride_step_num

        if self.does_use_global_img:
            global_process_data = self.transform(global_images)
            local_process_data = self.transform(local_images)
            process_data = torch.cat(
                [global_process_data, local_process_data], dim=0)
        else:
            process_data = self.transform(local_images)

        # one-hot-vector型に変換する
        one_hot_vector = np.zeros(12)
        one_hot_vector[record.label1] = 1
        if record.label2 != -1:
            one_hot_vector[record.label2] = 1
        if record.label3 != -1:
            one_hot_vector[record.label3] = 1
        if record.label4 != -1:
            one_hot_vector[record.label4] = 1

        return process_data, one_hot_vector

    def __len__(self):
        print('data_len={}'.format(self._data_len))
        return self._data_len
