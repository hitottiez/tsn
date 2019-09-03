import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
from numba import jit


def parse_args():
    parser = argparse.ArgumentParser(description='make fusion_tsn.txt from rgb_tsn.txt and flow_tsn.txt')
    parser.add_argument('--img_dir_path', help='/path/to/dataset/images/', required=True)
    parser.add_argument('--rgb_weight', type=float, default=1.0)
    parser.add_argument('--flow_weight', type=float, default=1.5)
    parser.add_argument('--worker', type=int, default=6)
    return parser.parse_args()


@jit(nopython=True)
def average(rgb_predicts, flow_predicts, rgb_weight, flow_weight):
    w1, w2 = rgb_weight, flow_weight
    w1 = w1 / (w1 + w2) if w1 != 0 and w2 != 0 else 0.5
    w2 = w2 / (w1 + w2) if w1 != 0 and w2 != 0 else 0.5
    return w1 * rgb_predicts + w2 * flow_predicts


def process(movie_name, args):
    print('process', movie_name)
    rgb_tsn_path = os.path.join(
        args.img_dir_path, movie_name, 'feature_results/rgb_tsn.txt')
    flow_tsn_path = os.path.join(
        args.img_dir_path, movie_name, 'feature_results/flow_tsn.txt')
    output_path = os.path.join(
        args.img_dir_path, movie_name, 'feature_results/fusion_tsn.txt')

    print('output:', output_path)

    rgb_f = open(rgb_tsn_path)
    flow_f = open(flow_tsn_path)
    with open(output_path, 'w') as out:
        for rgb_line in rgb_f:
            result = []
            rgb = json.loads(rgb_line)
            flow = json.loads(flow_f.readline())
            if len(rgb) == 0 or len(flow) == 0:
                out.write('[]\n')
                continue

            for i, flow_v in enumerate(flow):
                rgb_v = rgb[i]
                rgb_score = np.asarray(rgb_v['scores'])
                flow_score = np.asarray(flow_v['scores'])
                fusion_score = average(rgb_score, flow_score,
                                    args.rgb_weight, args.flow_weight)
                rgb_feature = np.asarray(rgb_v['before_fc_features'])
                flow_feature = np.asarray(flow_v['before_fc_features'])
                fusion_feature = np.hstack((rgb_feature, flow_feature))
                result.append({
                    'success': True,
                    'scores': fusion_score.tolist(),
                    'before_fc_features': fusion_feature.tolist()
                })
            out.write(json.dumps(result))
            out.write('\n')


def main():
    args = parse_args()
    movie_list = os.listdir(args.img_dir_path)
    print('weight', args.rgb_weight, args.flow_weight)
    process_args = []
    for movie_name in movie_list:
        process_args.append((movie_name, args))
    with Pool(args.worker) as pool:
        pool.starmap(process, process_args)


if __name__ == "__main__":
    main()
