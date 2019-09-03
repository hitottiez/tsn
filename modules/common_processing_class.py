# coding:utf-8

import os
import json

class Common(object):

    def __init__(self, img_dir_path=None, model_type=None, predict_function=None):
        self.img_dir_path = img_dir_path
        self.model_type = model_type
        self._predict_function = predict_function

    def bbox_file_loop(self):
        movie_list = os.listdir(self.img_dir_path)
        for movie_name in movie_list:

            # 出力先フォルダの作成
            save_dir = os.path.join(os.path.join(self.img_dir_path, movie_name), "feature_results")
            os.makedirs(save_dir, exist_ok=True)

            if self.model_type == "RGB":
                save_file_path = os.path.join(save_dir, "rgb_tsn.txt")
            elif self.model_type == "FLOW":
                save_file_path = os.path.join(save_dir, "flow_tsn.txt")
            elif self.model_type == "FUSION":
                save_file_path = os.path.join(save_dir, "fusion_tsn.txt")
            elif self.model_type == "CNN":
                save_file_path = os.path.join(save_dir, "cnn.txt")
            else:
                print("An unexpected model name has been set.")
                break

            print("feature extracting...", save_file_path)
            bbox_file_path = os.path.join(save_dir, 'det.txt')
            with open(bbox_file_path) as f:
                lines = f.readlines()
            with open(save_file_path, mode="w") as f:
                for line in lines:
                    image_name, bbox_array = line.strip().split('\.jpg'[1:])
                    img_folder_path = os.path.join(
                        self.img_dir_path, movie_name)
                    img_path = img_folder_path + "/" + image_name + ".jpg"
                    prev_img_path = img_folder_path + "/" + str(int(image_name) - 1) + ".jpg"

                    # det.txtのフォーマット変更に伴う修正
                    json_list = json.loads(bbox_array)
                    shape_bbox_list = []
                    for bbox in json_list:
                        box = bbox["box"]
                        shape_bbox_list.append(box)

                    print('processing {}.jpg'.format(image_name))

                    # skip feature extract when bbox is none
                    if len(shape_bbox_list) == 0:
                        f.write('[]\n')
                        continue

                    if self.model_type in ["RGB", "CNN"]:
                        with open(img_path, 'rb') as curr:
                            api_result = self._predict_function(curr, json.dumps(shape_bbox_list))
                            f.write(api_result.replace(' ', ''))
                    elif self.model_type in ["FLOW", "FUSION"]:
                        if image_name == "0":
                            f.write('[]\n')
                            continue
                        with open(img_path, 'rb') as curr, \
                                open(prev_img_path, 'rb') as prev:
                            api_result = self._predict_function(curr, prev,
                                                                json.dumps(shape_bbox_list))
                            f.write(api_result.replace(' ', ''))
                    f.write('\n')
