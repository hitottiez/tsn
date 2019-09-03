# coding:utf-8

import argparse
import json
import logging
import os
import traceback

from modules.common_processing_class import Common
from modules.image_processing import decode_image
from modules.util import (extract_flow_global_roi, extract_flow_local_roi,
                          extract_global_roi, extract_local_roi,
                          extract_predict)
from tsn import FlowExtractor, TSNActionDetector


class ActionDetectApplication:

    def __init__(self, model_type, weight_file, model_info):
        self.model_type = model_type
        self.weight_file = weight_file
        self.model_info = model_info
        self.flow_extractor = FlowExtractor()

    def model_initialize(self):
        """
        モデルの初期化
        """
        with open(self.model_info) as fjr:
            model_info_json_data = json.load(fjr)
            if self.model_type == "RGB":
                self._rgb_model = self.model_setup(self.weight_file, model_info_json_data)
                print("initialized RGB Predictor")
            if self.model_type == "FLOW":
                self._flow_model = self.model_setup(self.weight_file, model_info_json_data)
                print("initialized FLOW Predictor")

    def model_setup(self, weight_file, model_info):
        """
        モデルセットアップ
        """
        print(model_info)
        num_classes = model_info['num_classes']
        modality = model_info['modality']
        base_model = model_info['base_model']
        consensus_type = model_info['consensus_type']
        does_use_global_img = model_info['use_global_img']
        self.does_use_global_img_ = does_use_global_img

        return TSNActionDetector(weight_file, num_classes, modality,
                                 base_model=base_model,
                                 consensus_type=consensus_type,
                                 does_use_global_img=does_use_global_img)

    def rgb_predict(self, image_file, param):
        """
        RGB画像でTSN
        """
        image = decode_image(image_file)
        parameter = json.loads(param)
        try:
            roi_images, predicts, before_fc = self.__predict_rgb(image, parameter)
            result = extract_predict(predicts, roi_images, before_fc)
            result_encode_json = json.dumps(result)
            return result_encode_json
        except:
            traceback.print_exc()
            print("RGB TSN Error")

    def flow_predict(self, image_file, previous_image_file, param):
        """
        Flow画像でTSN
        """
        image = decode_image(image_file)
        prev_image = decode_image(previous_image_file)
        parameter = json.loads(param)
        try:
            flow_x, flow_y = self.flow_extractor.calc(prev_image, image)
            roi_images, predicts, before_fc = self.__predict_flow(flow_x, flow_y, parameter)
            result = extract_predict(predicts, roi_images, before_fc)
            result_encode_json = json.dumps(result)
            return result_encode_json
        except:
            traceback.print_exc()
            print("FLOW TSN Error")

    def decode_parameter(self, param):
        """
        パラメータをデコードする
        """
        try:
            data = json.loads(param.read())
            return data
        except Exception as e:
            print('Invalid Request Parameter[Invalid `parameter`.]')

    def __predict_rgb(self, image, boxes):
        """
        RGBで識別
        """
        roi_images = [extract_local_roi(image, box) for box in boxes]
        if self.does_use_global_img_:
            images = [extract_global_roi(image, box) for box in boxes]
            images.extend(roi_images)
        else:
            images = roi_images

        input_data = list(filter(lambda x: x is not None, images))
        predicts, before_fc = self._rgb_model.predict(input_data)
        return roi_images, predicts, before_fc

    def __predict_flow(self, flow_x, flow_y, boxes):
        """
        Opticalflowで識別
        """
        roi_images = [extract_flow_local_roi(flow_x, flow_y, box) for box in boxes]
        if self.does_use_global_img_:
            images = [extract_flow_global_roi(flow_x, flow_y, box) for box in boxes]
            images.extend(roi_images)
        else:
            images = roi_images
        input_data = list(filter(lambda x: x is not None, images))
        predicts, before_fc = self._flow_model.predict(input_data)
        return roi_images, predicts, before_fc


def main():
    # 引数まわり
    parser = argparse.ArgumentParser(description='TSN行動認識結果を取得するためのツール')
    parser.add_argument('--img_dir_path', help=' /path/to/dataset/images など')
    parser.add_argument('--modality', help='RGBまたはFLOWを指定')
    parser.add_argument('--weight_file', help='TSN学習時に出力された重みファイルを指定')
    parser.add_argument('--model_info', help='TSN学習時に出力されたモデル情報ファイルを指定')
    args = parser.parse_args()

    # ログフォルダの作成
    os.makedirs("./logs", exist_ok=True)

    tsn = ActionDetectApplication(model_type=args.modality,
                                  weight_file=args.weight_file,
                                  model_info=args.model_info)
    tsn.model_initialize()

    if args.modality == 'RGB':
        predict_function = tsn.rgb_predict
    elif args.modality == 'FLOW':
        predict_function = tsn.flow_predict

    common_instance = Common(args.img_dir_path, args.modality,
                             predict_function)
    common_instance.bbox_file_loop()


if __name__ == '__main__':
    main()
