import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

import _addpath
from vendors.tsn import transforms, dataset, ops
from vendors.tsn.transforms import *
from tsn import build_tsn
from ops import ConsensusModule
from train_modules.dataset import TSNDataSetCustom
from train_modules.dataset import TSNMultiDataSet

# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('num_class', type=int, default=100, help='Num Classes (NEW PARAMETER)')
parser.add_argument('test_list', type=str)
parser.add_argument('rgb_weights', type=str)
parser.add_argument('flow_weights', type=str)
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
parser.add_argument('--flow_prefix', type=str, default='')
parser.add_argument('--slide_num', type=int, default=1)
parser.add_argument('--stride_step_num', type=int, default=0)
parser.add_argument('--use_global_img', default=False, action='store_true')

args = parser.parse_args()


num_class = args.num_class
new_length = args.new_length


rgb_net = build_tsn(num_class, 1, 'RGB',
                    new_length=new_length,
                    base_model=args.arch,
                    consensus_type=args.crop_fusion_type,
                    dropout=args.dropout,
                    does_use_global_img=args.use_global_img)
flow_net = build_tsn(num_class, 1, 'Flow',
                     new_length=new_length,
                     base_model=args.arch,
                     consensus_type=args.crop_fusion_type,
                     dropout=args.dropout,
                     does_use_global_img=args.use_global_img)

checkpoint = torch.load(args.rgb_weights)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
rgb_net.load_state_dict(base_dict)

checkpoint = torch.load(args.flow_weights)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
flow_net.load_state_dict(base_dict)


if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(rgb_net.scale_size),
        GroupCenterCrop(rgb_net.input_size),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(rgb_net.input_size, rgb_net.scale_size)
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

rgb_transform = torchvision.transforms.Compose([
    cropping,
    Stack(roll=args.arch == 'BNInception'),
    ToTorchFormatTensor(div=args.arch != 'BNInception'),
    GroupNormalize(rgb_net.input_mean, rgb_net.input_std),
])

if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(flow_net.scale_size),
        GroupCenterCrop(flow_net.input_size),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(flow_net.input_size, flow_net.scale_size)
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

flow_transform = torchvision.transforms.Compose([
    cropping,
    Stack(roll=args.arch == 'BNInception'),
    ToTorchFormatTensor(div=args.arch != 'BNInception'),
    GroupNormalize(flow_net.input_mean, flow_net.input_std),
])


rgb_dataset = TSNDataSetCustom(args.test_list, num_segments=args.test_segments,
                               slide_num=args.slide_num,
                               stride_step_num=args.stride_step_num,
                               modality='RGB',
                               image_tmpl="{:d}.jpg",
                               does_use_global_img=args.use_global_img,
                               train_mode=False,
                               transform=rgb_transform
                               )
flow_dataset = TSNDataSetCustom(args.test_list, num_segments=args.test_segments,
                                slide_num=args.slide_num,
                                stride_step_num=args.stride_step_num,
                                modality='Flow',
                                image_tmpl="{:d}_{}.jpg",
                                does_use_global_img=args.use_global_img,
                                train_mode=False,
                                transform=flow_transform
                                )
dataset = TSNMultiDataSet(rgb_dataset, flow_dataset)
data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)

if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))


rgb_net = torch.nn.DataParallel(rgb_net.cuda(devices[0]), device_ids=devices)
rgb_net.eval()
flow_net = torch.nn.DataParallel(flow_net.cuda(devices[0]), device_ids=devices)
flow_net.eval()

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)
output = []


def eval_video(video_data):
    i, rgb_data, flow_data = video_data
    num_crop = args.test_crops
    rgb_length  = 3 * new_length
    flow_length = 2 * new_length

    # RGB Estimate
    data, label = rgb_data
    if args.use_global_img:
            # from global, global, global, local, local, local
            # to gloal, local, global, local, global, local
        input_parts = torch.split(data, rgb_length, dim=1)
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
            -1, rgb_length * 2, data.size(2), data.size(3)),
            volatile=True)
    else:
        input_var = torch.autograd.Variable(data.view(
            -1, rgb_length, data.size(2), data.size(3)),
            volatile=True)
    model_out, _ = rgb_net(input_var)
    rst_rgb = model_out.data.cpu().numpy().copy()

    # Flow Estimate
    data, label = flow_data
    if args.use_global_img:
            # from global, global, global, local, local, local
            # to gloal, local, global, local, global, local
        input_parts = torch.split(data, flow_length, dim=1)
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
            -1, flow_length * 2, data.size(2), data.size(3)),
            volatile=True)
    else:
        input_var = torch.autograd.Variable(data.view(
            -1, flow_length, data.size(2), data.size(3)),
            volatile=True)
    model_out, _ = flow_net(input_var)
    rst_flow = model_out.data.cpu().numpy().copy()

    # Fusion
    rgb_w = 1 / 2.5
    flow_w = 2 / 2.5
    rst = rgb_w * rst_rgb + flow_w * rst_flow

    return i, rst.reshape((num_crop, args.test_segments, num_class)).mean(axis=0).reshape(
        (args.test_segments, 1, num_class)
    ), label[0]

proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

for i, (data, label) in data_gen:
    if i >= max_num:
        break
    rst = eval_video((i, data, label))
    output.append(rst[1:])
    cnt_time = time.time() - proc_start_time
    print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                    total_num,
                                                                    float(cnt_time) / (i+1)))

conf_true = []
conf_pred = []
for x in output:
    # 上位何件のデータを対象とするかの判定
    video_labels = x[1]
    label_count = np.count_nonzero(video_labels == 1, axis=0)

    # 正解データのクラスインデックス番号を取得
    video_true = np.argwhere(video_labels == 1)
    video_true = video_true.tolist()[0]

    # predデータのクラスインデックス番号を取得
    unsorted_max_indices = np.mean(x[0], axis=0)
    unsorted_max_indices2 = np.argpartition(
        -unsorted_max_indices[0], label_count)[:label_count]
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
