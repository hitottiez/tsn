# -*- coding: utf-8 -*-
# 学習関係の便利クラス

import os
import json

def save_model_json(args, tsn):
    """
    モデルjsonファイルを作成する
    """
    data = {
        'num_classes': args.num_class,
        'modality': tsn.modality,
        'base_model': args.arch,
        'consensus_type': tsn.consensus_type,
        'new_length': tsn.new_length,
        'use_global_img': args.use_global_img
    }
    filepath = '_'.join((args.snapshot_pref, args.modality.lower(), 'model_info.json'))
    jsonstr = json.dumps(data)
    with open(filepath, 'w') as f:
        f.write(jsonstr)
