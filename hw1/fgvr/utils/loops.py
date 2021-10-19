# https://github.com/rwightman/pytorch-image-models/blob/b544ad4d3fcd02057ab9f43b118290f2a089566f/timm/utils/distributed.py#L11
# https://github.com/rwightman/pytorch-image-models/blob/master/train.py
from __future__ import print_function, division

import sys
import time
import torch
import numpy as np

from .misc_utils import AverageMeter, accuracy
from .dist_utils import reduce_tensor, distribute_bn

def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            if opt.distributed:
                reduced_loss = reduce_tensor(loss.data, opt.world_size)
                acc1 = reduce_tensor(acc1, opt.world_size)
                acc5 = reduce_tensor(acc5, opt.world_size)
            else:
                reduced_loss = loss.data

            losses.update(reduced_loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            if opt.local_rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, idx, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
                sys.stdout.flush()

    if opt.local_rank == 0:
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    if opt.distributed:
        distribute_bn(model, opt.world_size, True)

    return top1.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_cls = AverageMeter()
    losses_div = AverageMeter()
    losses_kd = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        out_s = model_s(input, classify_only=False)
        feat_s = out_s[:-1]
        logit_s = out_s[-1]

        with torch.no_grad():
            out_t = model_t(input, classify_only=False)
            feat_t = out_t[:-1]
            logit_t = out_t[-1]
            if opt.distill != 'ifacrd':
                feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'ifacrd':
            f_s = feat_s[-1]
            loss_kd = criterion_kd(f_s, feat_t)
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            if opt.distributed:
                reduced_loss = reduce_tensor(loss.data, opt.world_size)
                loss_cls = reduce_tensor(loss_cls.data, opt.world_size)
                loss_div = reduce_tensor(loss_div.data, opt.world_size)
                loss_kd = reduce_tensor(loss_kd.data, opt.world_size)
                acc1 = reduce_tensor(acc1, opt.world_size)
                acc5 = reduce_tensor(acc5, opt.world_size)
            else:
                reduced_loss = loss.data
                loss_cls = loss_cls.data
                loss_div = loss_div.data
                loss_kd = loss_kd.data

            losses.update(reduced_loss.item(), input.size(0))
            losses_cls.update(loss_cls.item(), input.size(0))
            losses_div.update(loss_div.item(), input.size(0))
            losses_kd.update(loss_kd.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            if opt.local_rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Loss cls {losses_cls.val:.4f} ({losses_cls.avg:.4f})\t'
                    'Loss div {losses_div.val:.4f} ({losses_div.avg:.4f})\t'
                    'Loss kd {losses_kd.val:.4f} ({losses_kd.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, idx, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, losses_cls=losses_cls,
                    losses_div=losses_div, losses_kd=losses_kd,
                    top1=top1, top5=top5))
                sys.stdout.flush()
    if opt.local_rank == 0:
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    if opt.distributed:
        distribute_bn(module_list, opt.world_size, True)

    return top1.avg, losses.avg, losses_cls.avg, losses_div.avg, losses_kd.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            torch.cuda.synchronize()

            if opt.distributed:
                reduced_loss = reduce_tensor(loss.data, opt.world_size)
                acc1 = reduce_tensor(acc1, opt.world_size)
                acc5 = reduce_tensor(acc5, opt.world_size)
            else:
                reduced_loss = loss.data

            losses.update(reduced_loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0 and opt.local_rank == 0:
                print('Val: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        if opt.local_rank == 0:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def feature_extraction(loader, backbone, opt):
    feature_vector = []
    labels_vector = []
    for idx, (x, y) in enumerate(loader):
        if torch.cuda.is_available():
            x = x.cuda()

        # get encoding
        with torch.no_grad():
            output = backbone(x, classify_only=False)
        features = output[-2].detach()

        feature_vector.extend(features.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if idx % opt.print_freq == 0 and opt.local_rank == 0:
            print(f"Step [{idx}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    if opt.local_rank == 0:
        print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector

