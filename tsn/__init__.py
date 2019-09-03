#-*- coding: utf-8 -*-

# TSNによる行動認識器
# ==============================================================================


from . import _addpath
from PIL import Image
from vendors import TSN_ROOT
from vendors.tsn.transforms import *
import torch
import os
import torchvision
from torch.autograd import Variable
import cv2
from .util import build_tsn, Transformer
from .flow import FlowExtractor


class TSNActionDetector:

    def __init__(self,
                 weightfile,
                 num_classes,
                 modality,
                 num_segments=1,
                 new_length=1,
                 base_model='BNInception',
                 consensus_type='avg',
                 dropout=0.7,
                 use_cuda=True,
                 does_use_global_img=False
                 ):
        self.num_classes = num_classes
        self.modality = modality
        self.num_segments = num_segments
        self.new_length = new_length
        self.base_model = base_model
        self.consensus_type = consensus_type
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.does_use_global_img = does_use_global_img

        self.net = self._build_tsn(weightfile)
        self.transform = Transformer(self.net, modality, base_model)

    def predict(self, images):
        """
        推論処理
        :param images: 人物領域を切り出した画像データ(切り出し画像単位で下記、作成する)
                       * modality=Flow: X方向、Y方向のオプティカルフロー画像のリストデータ[flow_x, flow_y]
                       * Otherwise: BGRの3チャンネル画像のリストデータ
        :return: 各人物の推論結果(スコア出力)
        """
        if len(images) < 1:
            return np.array([]), np.array([])
        if self.does_use_global_img:
            image_len = len(images) // 2
            global_images = images[0:image_len]
            local_images = images[image_len:]
            global_dataset = self.transform(global_images)
            local_dataset = self.transform(local_images)
            in_dataset = torch.cat([global_dataset, local_dataset], dim=1)
        else:
            in_dataset = self.transform(images)
        in_dataset = in_dataset.cuda() if self.use_cuda else in_dataset
        var = Variable(in_dataset, volatile=True)
        result, before_fc = self.net(var)
        return result.data.cpu().numpy(), before_fc.data.cpu().numpy()

    def _build_tsn(self, weightfile):
        """
        TSNモデルをビルドする (推論時はsoftmaxを使う)
        :param weightfile: モデルファイル
        :return: TSNのモデルオブジェクト
        """
        net = build_tsn(self.num_classes,
                        self.num_segments,
                        self.modality,
                        base_model=self.base_model,
                        new_length=self.new_length,
                        consensus_type=self.consensus_type,
                        before_softmax=True,
                        dropout=self.dropout,
                        does_use_global_img=self.does_use_global_img
                        )
        # 重みロード
        checkpoint = torch.load(weightfile)
        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(
            checkpoint['state_dict'].items())}
        net.load_state_dict(base_dict)
        if self.use_cuda:  # Use GPU
            net = net.cuda()
        net.eval()  # Evaluation Mode

        return net
