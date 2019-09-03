import datetime
import os
import shutil
import sys
import time

import numpy as np

import _addpath
import tensorboardX
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torchvision
from torch.nn.utils import clip_grad_norm
from train_modules.dataset import TSNDataSetCustom
from train_modules.train_opts import parser
from train_modules.train_util import save_model_json
from train_modules.transforms import GroupRandomVerticalFlip
from tsn import build_tsn
from vendors.tsn import dataset, transforms
from vendors.tsn.transforms import (GroupCenterCrop, GroupMultiScaleCrop,
                                    GroupNormalize, GroupOverSample,
                                    GroupRandomCrop, GroupRandomHorizontalFlip,
                                    GroupRandomSizedCrop, GroupScale,
                                    IdentityTransform, Stack,
                                    ToTorchFormatTensor)

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.new_length = 1  # 1固定

    model = build_tsn(args.num_class, args.num_segments, args.modality,
                        new_length=args.new_length,
                        base_model=args.arch,
                        consensus_type=args.consensus_type,
                        dropout=args.dropout,
                        partial_bn=not args.no_partialbn,
                        does_use_global_img=args.use_global_img)
    policies = model.get_optim_policies()

    save_model_json(args, model)
    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    input_size = model.input_size

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    # if args.modality == 'RGB':
    #     data_length = 1
    # elif args.modality in ['Flow', 'RGBDiff']:
    #     data_length = 5

    image_tmpl = "{:d}.jpg" if args.modality in [
        "RGB", "RGBDiff"] else "{:d}_{}.jpg"
    is_flow = args.modality == 'Flow'
    train_loader = torch.utils.data.DataLoader(
        TSNDataSetCustom(args.train_list, num_segments=args.num_segments,
                         slide_num=args.slide_num,
                         stride_step_num=args.stride_step_num,
                         modality=args.modality,
                         image_tmpl=image_tmpl,
                         does_use_global_img=args.use_global_img,
                         transform=torchvision.transforms.Compose([
                             GroupScale(input_size),
                             GroupRandomVerticalFlip(is_flow=is_flow),
                             GroupRandomHorizontalFlip(is_flow=is_flow),
                             Stack(roll=args.arch == 'BNInception'),
                             ToTorchFormatTensor(
                                 div=args.arch != 'BNInception'),
                             normalize,
                         ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSetCustom(args.val_list, num_segments=args.num_segments,
                         slide_num=args.slide_num,
                         stride_step_num=args.stride_step_num,
                         modality=args.modality,
                         image_tmpl=image_tmpl,
                         does_use_global_img=args.use_global_img,
                         train_mode=False,
                         transform=torchvision.transforms.Compose([
                             GroupScale(int(scale_size)),
                             GroupCenterCrop(crop_size),
                             Stack(roll=args.arch == 'BNInception'),
                             ToTorchFormatTensor(
                                 div=args.arch != 'BNInception'),
                             normalize,
                         ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        #criterion = torch.nn.CrossEntropyLoss().cuda()
        criterion = torch.nn.BCEWithLogitsLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    tensorborardx_logdir = './tensorboardx_logs/' +datetime.datetime.now().strftime('%Y%m%d_%H%M')
    os.makedirs(tensorborardx_logdir, exist_ok=True)
    summaryWriter = tensorboardX.SummaryWriter(tensorborardx_logdir)
    for epoch in range(args.start_epoch, args.epochs):
        sys.stdout.flush()
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        tr_prec1, tr_loss1 = train(
            train_loader, model, criterion, optimizer, epoch, summaryWriter)
        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1, loss1 = validate(
                val_loader, model, criterion, (epoch + 1) * len(train_loader))
            summaryWriter.add_scalars('epoch/prec',
                                      {'train': tr_prec1, 'test': prec1},
                                      (epoch + 1))
            summaryWriter.add_scalars('epoch/loss',
                                      {'train': tr_loss1, 'test': loss1},
                                      (epoch + 1))

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)
    summaryWriter.close()


def train(train_loader, model, criterion, optimizer, epoch, summaryWriter):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top = AverageMeter()
    #top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    offset_compute_count = (epoch) * len(train_loader) + 1
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = torch.FloatTensor(np.array(target.numpy(),np.long))
        target = target.cuda(async=True)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output, _ = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec = accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))
        top.update(prec, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # tensorboard用ログ出力
        summaryWriter.add_scalar('data/loss', loss, offset_compute_count + i)
        summaryWriter.add_scalar('data/prec1', top.val, offset_compute_count + i)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top.val:.3f} ({top.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top=top, lr=optimizer.param_groups[-1]['lr'])))
    return top.avg, losses.avg


def validate(val_loader, model, criterion, iter, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = torch.FloatTensor(np.array(target.numpy(),np.long))
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output, _ = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec = accuracy(output.data, target)

        losses.update(loss.data[0], input.size(0))
        top.update(prec, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top.val:.3f} ({top.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top=top)))

    print(('Testing Results: Prec@1 {top.avg:.3f} Loss {loss.avg:.5f}'
          .format(top=top, loss=losses)))

    return top.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, args.modality.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    topk = (1,4)
    maxk = max(topk)
    batch_size = target.size(0)

    # 上位何件のデータを参照するかターゲットデータのラベル数を参照
    label_count = np.count_nonzero(target == 1, axis = 1)

    # one-hot配列からクラス番号を取得
    class_index_array, class_num_array = np.argwhere(target == 1)

    correct_rate_array = []
    for idx in range(batch_size):
        # バッチごとにつけられている行動数分の上位ラベルデータを取得
        acc, pred = output.topk(int(label_count[idx]), 1, True, True)

        row = target[idx]
        class_label_array = np.where(row == 1)[0]
 
        pre_label_array = []
        for pre_row in pred.t(): #?x8を8x?に変更してループ
            pre_label_array.append(pre_row[idx])

        # 重複する要素の個数を取得
        duplicate_num = len(list(set(class_label_array) & set(pre_label_array)))
        
        # 1バッチごとに正答率を算出
        correct_rate = duplicate_num / len(pre_label_array)
        correct_rate_array.append(correct_rate)

    correct_rate_all = 0
    for rate in correct_rate_array:
        correct_rate_all += rate
    res = 100 * (correct_rate_all / batch_size)
    return res


if __name__ == '__main__':
    main()
