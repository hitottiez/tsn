# tsn

このリポジトリは、[オリジナル](https://github.com/yjxiong/tsn-pytorch)をベースとし、人物画像と全体画像を使用して学習するソースを含みます。

## クローン＆Dockerビルド

```
git clone  --recursive https://github.com/hitottiez/tsn.git
cd tsn
docker build -t <tagname> .
```

## Docker起動&コンテナログイン&セットアップ

もしデータセットを特定のディレクトリで管理している場合は、以下のようにマウントして下さい。

```
docker run -d -it --name <container_name> \
    --mount type=bind,src=/<path/to/tsn>/,dst=/opt/multi_actrecog/tsn \
    --mount type=bind,src=/<path/to/dataset>/,dst=/mnt/dataset \
    <image name> /bin/bash
```

正常に起動したら、以下のコマンドでログインします。

```
docker exec -it <container_name> /bin/bash
```

## 学習用データセット作成

以降の作業を行う前に、データセットを決められたディレクトリ構成で設置しておく必要があります。  
詳細は[mht-paf](https://github.com/hitottiez/mht-paf)を参照して下さい。

学習用データ作成
```
cd training_tools
python create_labels_for_tsn_train.py \
    --label_dir /mnt/dataset/okutama_action_dataset/okutama_3840_2160/multi_labels/train/ \
    --output_dir /mnt/dataset/okutama_action_dataset/okutama_3840_2160/labels_tsn/train \
    --min_range 10
```

テスト用データ作成
```
cd training_tools
python create_labels_for_tsn_train.py \
    --label_dir /mnt/dataset/okutama_action_dataset/okutama_3840_2160/multi_labels/test/ \
    --output_dir /mnt/dataset/okutama_action_dataset/okutama_3840_2160/labels_tsn/test \
    --min_range 10
```

`--output_dir`は`labels`と同じディレクトリに設置して下さい。

## Flow画像作成

もしFlowモデルを学習したい場合は、以下のコマンドでRGB画像からFlow画像を生成します。

```
cd training_tools
python create_flow_image.py \
    --input_dirpath /mnt/dataset/okutama_action_dataset/okutama_3840_2160/
    --worker 8
```

## 学習

train.py

```
cd training_tools
python3 -u train.py \
    12 \
    RGB \
    /mnt/dataset/okutama_action_dataset/okutama_3840_2160/labels_tsn/train/datalist.txt \
    /mnt/dataset/okutama_action_dataset/okutama_3840_2160/labels_tsn/test/datalist.txt \
    --arch resnet101 \
    --num_segments 3 \
    --slide_num 5 \
    --stride_step_num 10 \
    --gd 20 \
    --lr 0.001 \
    --lr_steps 30 60 \
    --epochs 80 \
    -b 64 \
    -j 0 \
    --dropout 0.8 \
    --eval-freq 2 \
    --use_global_img \
    --snapshot_pref base_resnet101
```


## 評価

以下はRGBモデルを評価する例です。

```
cd training_tools
python3 -u test.py \
    12 \
    RGB \
    /mnt/dataset/okutama_action_dataset/okutama_3840_2160/labels_tsn/test/datalist.txt \
    ./base_resnet101_rgb_model_best.pth.tar \
    --arch resnet101 \
    --test_segments 2 \
    --slide_num 3 \
    --stride_step_num 10 \
    --test_crops 1 \
    -j 1 \
    --dropout 0.7 \
    --use_global_img \
    --save_scores rgb_score.npz
```

モデルに一致するようにmodality、`--use_global_img`を指定して下さい。
さもなければ以下のようなエラーが出ます。

RGB, Flowの間違い
```
RuntimeError: While copying the parameter named XXX, whose dimensions in the model are torch.Size([...]) and whose dimensions in the checkpoint are torch.Size([...]).
```

`--use_global_img`の間違い
```
KeyError: 'unexpected key "XXX" in state_dict'
```


RGB+FLOWの評価をする場合は、`fusion_test.py`を実行します。
```
cd training_tools
python3 -u fusion_test.py \
    12 \
    /mnt/dataset/okutama_action_dataset/okutama_3840_2160/labels_tsn/test/datalist.txt \
    ./base_resnet101_rgb_model_best.pth.tar \
    ./base_resnet101_flow_model_best.pth.tar \
    --arch resnet101 \
    --test_segments 2 \
    --slide_num 3 \
    --stride_step_num 10 \
    --test_crops 1 \
    -j 1 \
    --dropout 0.7 \
    --use_global_img \
    --save_scores fusion_score.npz

```

# RGB, FLOWの特徴量ファイル

```
python tsn_action_recognition.py \
    --img_dir_path /mnt/dataset/okutama_action_dataset/okutama_3840_2160/images \
    --modality RGB \
    --weight_file base_resnet101_rgb_model_best.pth.tar \
    --model_info base_resnet101_rgb_model_info.json
```

```
python tsn_action_recognition.py \
    --img_dir_path /mnt/dataset/okutama_action_dataset/okutama_3840_2160/images \
    --modality FLOW \
    --weight_file base_resnet101_flow_model_best.pth.tar \
    --model_info base_resnet101_flow_model_info.json
```

# RGB + FLOWの特徴量ファイル

RGB, FLOWの特徴量ファイルを作成した後に、以下を実行して下さい。
```
python tsn_fusion_maker.py \
    --img_dir_path /mnt/dataset/okutama_action_dataset/okutama_3840_2160/images/ \
    --rgb_weight 1.0 \
    --flow_weight 1.5 \
    --worker 8
```
