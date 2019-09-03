import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

import _addpath
from vendors.tsn import transforms, dataset, ops
from tsn import build_tsn
from train_modules.dataset import TSNDataSetCustom
from vendors.tsn.transforms import *
from vendors.tsn.ops import ConsensusModule

# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('num_class', type=int, default=100, help='Num Classes (NEW PARAMETER)')
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--new_length', type=int, default=1, help='Model Length (NEW PARAMETER)')
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--slide_num', type=int, default=1)
parser.add_argument('--stride_step_num', type=int, default=0)
parser.add_argument('--use_global_img', default=False, action='store_true')

args = parser.parse_args()


num_class = args.num_class
if args.new_length:
    new_length = args.new_length
else:
    new_length = 1 if args.modality == 'RGB' else 5


net = build_tsn(num_class, 1, args.modality,
          new_length=new_length,
          base_model=args.arch,
          consensus_type=args.crop_fusion_type,
          dropout=args.dropout,
          does_use_global_img=args.use_global_img)

checkpoint = torch.load(args.weights)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)

if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        # サイズバラバラの複数画像を切り出せるようにサイズを合せてるかも？
        GroupScale(net.scale_size),
        # 例えば320x240だったら、中心から244x244を切り出している
        GroupCenterCrop(net.input_size),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        # 画像から10個切り出している
        GroupOverSample(net.input_size, net.scale_size)
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

data_loader = torch.utils.data.DataLoader(
    TSNDataSetCustom(args.test_list, num_segments=args.test_segments,
                     slide_num=args.slide_num,
                     stride_step_num=args.stride_step_num,
                     modality=args.modality,
                     image_tmpl="{:d}.jpg" if args.modality in ['RGB', 'RGBDiff'] else "{:d}_{}.jpg",
                     does_use_global_img=args.use_global_img,
                     train_mode=False,
                     # Composeは最初の要素から次々に処理する
                     # つまり、前処理をやっていく順番通りにリストを作る
                     transform=torchvision.transforms.Compose([
                         cropping,
                         Stack(roll=args.arch == 'BNInception'),
                         ToTorchFormatTensor(div=args.arch != 'BNInception'),
                         GroupNormalize(net.input_mean, net.input_std),
                     ])),
    batch_size=1, shuffle=False,
    num_workers=args.workers * 2, pin_memory=True)

if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))


net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
net.eval()

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)
output = []


def eval_video(video_data):
    # i: データ番号
    # data: num_segment分のデータ(1, 3xnum_segmentxtest_crop, height, width)
    # label: 正解ラベル(1次元のTorch)
    i, data, label = video_data
    num_crop = args.test_crops

    if args.modality == 'RGB':
        length = 3 * new_length
    elif args.modality == 'Flow':
        length = 2 * new_length
    elif args.modality == 'RGBDiff':
        length = 3 * (new_length + 1)
    else:
        raise ValueError("Unknown modality " + args.modality)

    if args.use_global_img:
        # from global, global, global, local, local, local
        # to gloal, local, global, local, global, local
        input_parts = torch.split(data, length, dim=1)
        original_image_num = len(input_parts) // 2
        data = None
        for i in range(original_image_num):
            tmp = torch.cat(
                [input_parts[i], input_parts[original_image_num + i]],
                dim=0)
            if data is None:
                data = tmp
            else:
                data = torch.cat([data, tmp], dim=0)

        input_var = torch.autograd.Variable(data.view(
            -1, length * 2, data.size(2), data.size(3)),
                                            volatile=True)
    else:
        input_var = torch.autograd.Variable(data.view(
            -1, length, data.size(2), data.size(3)),
            volatile=True)
    # shape(バッチサイズxセグメント数xcrop数, 10（行動ラベル）)
    model_out, _ = net(input_var)
    rst = model_out.data.cpu().numpy().copy()

    return i, rst.reshape((num_crop, args.test_segments, num_class)).mean(axis=0).reshape(
        (args.test_segments, 1, num_class)
    ), label[0]


proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

for i, (data, label) in data_gen:
    if i >= max_num:
        break
    rst = eval_video((i, data, label))

    # 結果(segment, 1, 10(行動))と正解ラベルを追加
    # 0番目の要素はデータ番号なのでいらない
    output.append(rst[1:])
    cnt_time = time.time() - proc_start_time
    print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                    total_num,
                                                                    float(cnt_time) / (i+1)))

# segment数で結果(segment, 1, 10(行動))の平均とって、（10行動分の評価結果）
# そのうち最も
# 値の大きいものを結果にしている
#video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]
conf_true = []
conf_pred = []
for x in output:
    # 上位何件のデータを対象とするかの判定
    video_labels = x[1]
    label_count = np.count_nonzero(video_labels == 1, axis = 0)

    # 正解データのクラスインデックス番号を取得
    video_true = np.argwhere(video_labels == 1)
    video_true = video_true.tolist()[0]

    # predデータのクラスインデックス番号を取得
    unsorted_max_indices = np.mean(x[0], axis=0)
    unsorted_max_indices2 = np.argpartition(-unsorted_max_indices[0], label_count)[:label_count]
    y = unsorted_max_indices[0][unsorted_max_indices2]
    indices = np.argsort(-y)
    video_pred = unsorted_max_indices2[indices]
    video_pred = video_pred.tolist()

    for true in video_true:
        for pred in video_pred:
            if true == pred:
                # 元の配列から一致した要素を取り除きマトリックス用の配列に一致した要素を追加
                video_true.remove(true)
                video_pred.remove(true)
                conf_true.append(true)
                conf_pred.append(true)
    # 取り除いた配列の要素をマトリックス用の配列に追加
    conf_true.extend(video_true)
    conf_pred.extend(video_pred)            

cf = confusion_matrix(conf_true, conf_pred).astype(float)

cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit / cls_cnt

print(cf)
print(cls_acc)
print(cls_hit)

print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))

if args.save_scores is not None:
    np.savez(args.save_scores, preds=conf_pred, labels=conf_true)
