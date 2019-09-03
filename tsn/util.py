#-*- coding: utf-8 -*-

# TSN関連、ユーティリティ関数
# ==============================================================================

from PIL import Image
from vendors import TSN_ROOT
from .models import TSNCustom
from vendors.tsn.transforms import *
import torch
import os
import torchvision
from torch.autograd import Variable
import cv2

def build_tsn(num_class,
              num_segments,
              modality,
              base_model='resnet101',
              new_length=None,
              consensus_type='avg',
              before_softmax=True,
              dropout=0.8,
              crop_num=1,
              partial_bn=True,
              does_use_global_img=False):
    """
    TSNのネットワークのインスタンスを作成して返却 (パラメータは元と同じ)
    ※役割 TSNクラスは内部処理でベースモデルをロードしている。この時、モデルファイルのベースパスが
      TSN_ROOTからのルートディレクトリであるため、カレントディレクトリをTSN_ROOTに移動してからTSNクラスを読み込む必要がある。
      つまり、この関数は「一旦TSN_ROOTにカレントディレクトリを変えてから、TSNオブジェクトを作成してまたカレントディレクトリを元に戻す」処理を行っている。
    """
    current_dir = os.path.abspath(os.getcwd()) # 現在のディレクトリ
    os.chdir(TSN_ROOT)
    tsn = TSNCustom(num_class,
                    num_segments,
                    modality,
                    base_model=base_model,
                    new_length=new_length,
                    consensus_type=consensus_type,
                    before_softmax=before_softmax,
                    dropout=dropout,
                    crop_num=crop_num,
                    partial_bn=partial_bn,
                    does_use_global_img=does_use_global_img)
    os.chdir(current_dir)
    return tsn

class Transformer:
    """
    画像データをモデルに入力するデータに変換するクラス
    """

    def __init__(self, net, modality, base_model):
        self.modality = modality
        self.base_model = base_model
        transform = self._build_transform(net, modality, base_model)
        self.mapper = TransformMapper(modality, transform)

    def __call__(self, images):
        """
        入力データに変換する
        :param images: 人物領域を切り出した画像データ(切り出し画像単位で下記、作成する)
                       * modality=Flow: X方向、Y方向のオプティカルフロー画像のリストデータ[flow_x, flow_y]
                       * Otherwise: BGRの3チャンネル画像のリストデータ
        :return: 各人物の推論結果(スコア出力)
        """
        indata = list(map(self.mapper, images))
        return torch.stack(indata)

    def _build_transform(self, net, modality, base_model):
        groupScale = GroupScale(int(net.scale_size))
        groupCenterCrop = GroupCenterCrop(net.crop_size)
        stack = Stack(roll=base_model=='BNInception')
        toTorchFormatTensor = ToTorchFormatTensor(div=base_model != 'BNInception')
        if self.modality == 'RGBDiff':
            normalize = IdentityTransform()
        else:
            normalize = GroupNormalize(net.input_mean, net.input_std)

        return torchvision.transforms.Compose([
            groupScale,
            groupCenterCrop,
            stack,
            toTorchFormatTensor,
            normalize
        ])

class TransformMapper:
    """
    Transformerが使用するmapperクラス
    """

    def __init__(self, modality, transform):
        self.modality = modality
        self.transform = transform

    def __call__(self, input):
        # modalityがRGBの場合、BGRをRGBに変換する
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            rgb_image = cv2.cvtColor(input, cv2.COLOR_BGR2RGB) # BGR to RGB
            output = Image.fromarray(rgb_image).convert('RGB')
            return self.transform([output])
        else:
            flow_x, flow_y = input
            if len(flow_x.shape) > 2:
                flow_x = cv2.cvtColor(flow_x, cv2.COLOR_BGR2GRAY)
            if len(flow_y.shape) > 2:
                flow_y = cv2.cvtColor(flow_y, cv2.COLOR_BGR2GRAY)
            flow_x = Image.fromarray(flow_x).convert('L')
            flow_y = Image.fromarray(flow_y).convert('L')
            output = [flow_x, flow_y]
            return self.transform(output)
