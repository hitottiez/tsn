# オプティカルフロー作成
# ==============================================================================

import py_tvl1flow
from py_tvl1flow import TVL1Flow

class FlowExtractor:

    def __init__(self):
        self.extractor = TVL1Flow()

    def calc(self, previous_image, image):
        """
        BGR画像を受け取り、1チャンネルのx, y画像を出力する
        :param previous_image: １つ前のフレーム画像
        :param image: 現在のフレーム画像
        """
        op_image = self.extractor.calc(previous_image, image)
        h, w = op_image.shape[:-1]
        flow_x = op_image[:, :, 0]
        flow_y = op_image[:, :, 1]

        return flow_x, flow_y

    @classmethod
    def set_gpu(cls, gpu_id):
        """
        GPUを指定する
        """
        py_tvl1flow.setDevice(gpu_id)
