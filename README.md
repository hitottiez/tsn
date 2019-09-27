# tsn
This repository includes modified code of [tsn-pytorch](https://github.com/yjxiong/tsn-pytorch) and it can utilize the local and global cropped images.


## Cloning codes and building a docker image
```
git clone  --recursive https://github.com/hitottiez/tsn.git
cd tsn
docker build -t <tagname> .
```

## Runnning a docker container and login
Run:
```
docker run -d -it --name <container_name> \
    --mount type=bind,src=/<path/to/tsn>/,dst=/opt/multi_actrecog/tsn \
    --mount type=bind,src=/<path/to/dataset>/,dst=/mnt/dataset \
    <image name> /bin/bash
```

Login:
```
docker exec -it <container_name> /bin/bash
```

## Making dataset
Put the dataset in the directory which has proper structure (refer [mht-paf](https://github.com/hitottiez/mht-paf)).

Make training dataset:
```
cd training_tools
python create_labels_for_tsn_train.py \
    --label_dir /mnt/dataset/okutama_action_dataset/okutama_3840_2160/multi_labels/train/ \
    --output_dir /mnt/dataset/okutama_action_dataset/okutama_3840_2160/labels_tsn/train \
    --min_range 10
```

Make test dataset:
```
cd training_tools
python create_labels_for_tsn_train.py \
    --label_dir /mnt/dataset/okutama_action_dataset/okutama_3840_2160/multi_labels/test/ \
    --output_dir /mnt/dataset/okutama_action_dataset/okutama_3840_2160/labels_tsn/test \
    --min_range 10
```
`--output_dir` needs to be set in the same directory as `labels`.

## Making Flow images
Make Flow images from RGB images:
```
cd training_tools
python create_flow_image.py \
    --input_dirpath /mnt/dataset/okutama_action_dataset/okutama_3840_2160/
    --worker 8
```

## Training the primitive action feature extraction model
Run:
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

## Evaluation of action recognition
Example of RGB model:
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
Set modality and `--use_global_img` as corresponding to the model.

Error: RGB, Flow
```
RuntimeError: While copying the parameter named XXX, whose dimensions in the model are torch.Size([...]) and whose dimensions in the checkpoint are torch.Size([...]).
```

Error: `--use_global_img`
```
KeyError: 'unexpected key "XXX" in state_dict'
```


Evaluate RGB+Flow:
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

## Feature extraction of RGB/Flow
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

# Feature extraction of RGB+Flow
After making files of RGB and Flow featuers above and run:
```
python tsn_fusion_maker.py \
    --img_dir_path /mnt/dataset/okutama_action_dataset/okutama_3840_2160/images/ \
    --rgb_weight 1.0 \
    --flow_weight 1.5 \
    --worker 8
```
