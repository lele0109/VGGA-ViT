# vgg预训练imagenet,后接AUS-Net结构,预训练backbone参与训练,交叉验证

from __future__ import print_function
import numpy as np
import os
import sys
import time
from numpy.core.arrayprint import format_float_positional
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
import argparse
import socket
import torch.multiprocessing as mp
import torch.distributed as dist
import csv
import tensorboard_logger as tb_logger

from torchvision import transforms, datasets
from util import adjust_learning_rate, AverageMeter

sys.path.append("..")


def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=32, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=60, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='25,50,75,90',
                        help='where to decay lr, can be a list')  # '15,30,45,60,75'
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')

    # model definition
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'resnet50x2', 'resnet50x4', 'VGG16', 'VGG16AUS', 'VGG16_', 'vggvit'])
    parser.add_argument('--model_path', type=str, default=None, help='the model to test')
    parser.add_argument('--layer', type=int, default=6, help='which layer to evaluate')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        help='model architecture (default: resnet18)')
    # crop
    parser.add_argument('--crop', type=float, default=0.2, help='minimum crop')

    # dataset
    parser.add_argument('--dataset', type=str, default='imagenet100', choices=['imagenet100', 'imagenet', 'ruxian'])

    # resume
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # augmentation
    parser.add_argument('--aug', type=str, default='CJ', choices=['NULL', 'CJ'])

    # add BN
    parser.add_argument('--bn', action='store_true', help='use parameter-free BN')
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')

    # warmup
    parser.add_argument('--warm', action='store_true', help='add warm-up setting')
    parser.add_argument('--amp', action='store_true', help='using mixed precision')
    parser.add_argument('--opt_level', type=str, default='O2', choices=['O1', 'O2'])
    parser.add_argument('--syncBN', action='store_true', help='enable synchronized BN')

    # GPU setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    opt = parser.parse_args()

    opt.data_folder = '/home/hdd/lhy/CMC-master/result/{}/shuffle_B'.format(opt.dataset)
    opt.save_path = '/home/hdd/lhy/CMC-master/result/{}_save/shuffle_B'.format(opt.dataset)
    opt.tb_path = '/home/hdd/lhy/CMC-master/result/{}_tensorboard/shuffle_B'.format(opt.dataset)

    if opt.dataset == 'imagenet':
        if 'alexnet' not in opt.model:
            opt.crop = 0.08

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'vggvit_shuffleB_64→2_digitarg'
    opt.model_name = '{}_bsz_{}_lr_{}_decay_{}_crop_{}'.format(opt.model_name, opt.batch_size, opt.learning_rate,
                                                               opt.weight_decay, opt.crop)

    if opt.amp:
        opt.model_name = '{}_amp_{}'.format(opt.model_name, opt.opt_level)

    opt.model_name = '{}_aug_{}'.format(opt.model_name, opt.aug)

    if opt.bn:
        opt.model_name = '{}_useBN'.format(opt.model_name)
    if opt.adam:
        opt.model_name = '{}_useAdam'.format(opt.model_name)
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name + '_layer{}'.format(opt.layer))
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.save_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'imagenet100':
        opt.n_label = 100
    if opt.dataset == 'imagenet':
        opt.n_label = 1000
    if opt.dataset == 'ruxian':
        opt.n_label = 2

    return opt


def main():
    global best_acc1
    best_acc1 = 0

    args = parse_option()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    for fold in range(5):
        print('正在训练第' + str(fold) + '折')
        # set the data loader
        train_folder = os.path.join(args.data_folder, 'train{}'.format(fold))
        val_folder = os.path.join(args.data_folder, 'val{}'.format(fold))

        image_size = 224
        crop_padding = 32
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)

        if args.aug == 'NULL':
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(args.crop, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif args.aug == 'CJ':
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(args.crop, 1.)),
                transforms.CenterCrop(image_size),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                # transforms.RandomRotation(degrees=(-10,10)),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            raise NotImplemented('augmentation not supported: {}'.format(args.aug))

        train_dataset = datasets.ImageFolder(train_folder, train_transform)
        val_dataset = datasets.ImageFolder(
            val_folder,
            transforms.Compose([
                transforms.Resize(image_size),
                # transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])
        )

        print(len(train_dataset))
        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
        # create model and optimizer
        if args.model == 'vggvit':
            from models.model import sdlVggformer
            model = sdlVggformer()
        else:
            raise NotImplementedError('model not supported {}'.format(args.model))

        print('==> no loading pre-trained model')

        print('==> done')

        model = model.cuda()

        criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate,
                                     # betas=(args.beta1, args.beta2),
                                     # weight_decay=args.weight_decay,
                                     eps=1e-8)

        model.train()
        cudnn.benchmark = True

        # optionally resume from a checkpoint
        args.start_epoch = 1
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location='cpu')
                # checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch'] + 1
                optimizer.load_state_dict(checkpoint['optimizer'])
                best_acc1 = checkpoint['best_acc1']
                best_acc1 = best_acc1.cuda()
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                if 'opt' in checkpoint.keys():
                    # resume optimization hyper-parameters
                    print('=> resume hyper parameters')
                    if 'bn' in vars(checkpoint['opt']):
                        print('using bn: ', checkpoint['opt'].bn)
                    if 'adam' in vars(checkpoint['opt']):
                        print('using adam: ', checkpoint['opt'].adam)
                    if 'cosine' in vars(checkpoint['opt']):
                        print('using cosine: ', checkpoint['opt'].cosine)
                    args.learning_rate = checkpoint['opt'].learning_rate
                    # args.lr_decay_epochs = checkpoint['opt'].lr_decay_epochs
                    args.lr_decay_rate = checkpoint['opt'].lr_decay_rate
                    args.momentum = checkpoint['opt'].momentum
                    args.weight_decay = checkpoint['opt'].weight_decay
                    args.beta1 = checkpoint['opt'].beta1
                    args.beta2 = checkpoint['opt'].beta2
                del checkpoint
                torch.cuda.empty_cache()
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        # set cosine annealing scheduler
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3) * 0.1
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min, -1)
            # dummy loop to catch up with current epoch
            for i in range(1, args.start_epoch):
                scheduler.step()

        logger = tb_logger.Logger(logdir=args.tb_folder + '/{}'.format(fold), flush_secs=2)

        # routine
        for epoch in range(args.start_epoch, args.epochs + 1):

            if args.cosine:
                scheduler.step()
            else:
                adjust_learning_rate(epoch, args, optimizer)
            print("==> training...")

            time1 = time.time()
            train_acc, train_acc5, train_loss = train(epoch, train_loader, model, criterion, optimizer, args)
            time2 = time.time()
            print('train epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            logger.log_value('train_acc', train_acc, epoch)
            logger.log_value('train_acc5', train_acc5, epoch)
            logger.log_value('train_loss', train_loss, epoch)
            logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            print("==> testing...")
            test_acc, test_acc5, test_loss = validate(val_loader, model, criterion, args)

            logger.log_value('test_acc', test_acc, epoch)
            logger.log_value('test_acc5', test_acc5, epoch)
            logger.log_value('test_loss', test_loss, epoch)

            # save the best model
            if test_acc > best_acc1:
                best_acc1 = test_acc
                state = {
                    # 'opt': args,
                    # 'epoch': epoch,
                    'model': model.state_dict(),
                    # 'best_acc1': best_acc1,
                    # 'optimizer': optimizer.state_dict(),
                }
                save_name = '{}_layer{}_fold{}.pth'.format(args.model, args.layer, fold)
                save_name = os.path.join(args.save_folder, save_name)
                print('saving best model!')
                torch.save(state, save_name)
                pass
        acc_name = os.path.join(args.save_folder, 'fold_{}_acc_best.txt'.format(fold))
        file = open(acc_name, 'w')
        file.write(str(best_acc1))
        best_acc1 = 0
        # csv验证输出
        ckpt_model = torch.load(save_name)
        model.load_state_dict(ckpt_model['model'])
        model = model.cuda()
        model.eval()
        feature_list = []
        target_list = []
        csv_name = os.path.join(args.save_folder, 'model_pre_fold{}.csv'.format(fold))
        g = open(csv_name, 'w')
        with torch.no_grad():
            # end = time.time()
            for idx, (input, target) in enumerate(test_loader):
                input = input.float()
                if args.gpu is not None:
                    input = input.cuda(args.gpu, non_blocking=True)
                input = input.float()
                input.unsqueeze(0)
                input = input.cuda()
                target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                feat = model(input)  # moco,insdis AUSnet
                feat = feat.detach()
                acc1, acc5 = accuracy(feat, target, topk=(1, 2))
                print(acc1)
                maxk = 1
                _, pred = feat.topk(maxk, 1, True, True)
                pred = pred.t()
                target = target.cuda().data.cpu().numpy()
                feat = pred.cuda().data.cpu().numpy()

                # print(feat)
                if (idx == 0):
                    feature_list = feat
                    target_list = target
                else:
                    feature_list = np.concatenate((feature_list, feat), axis=0)
                    target_list = np.concatenate((target_list, target), axis=0)
                writer = csv.writer(g)
                writer.writerow([idx, feature_list[idx], target_list[idx]])
        g.close
        torch.cuda.empty_cache()


def set_lr(optimizer, lr):
    """
    set the learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch, train_loader, model, criterion, optimizer, opt):
    """
    one epoch training
    """

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if opt.gpu is not None:
            input = input.cuda(opt.gpu, non_blocking=True)
        input = input.cuda().float()

        target = target.cuda(opt.gpu, non_blocking=True)

        # ===================forward=====================
        # with torch.no_grad():
        feat = model(input).cuda()
        output = feat
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top3.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top3.val:.3f} ({top3.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top3=top3))
            sys.stdout.flush()

    return top1.avg, top3.avg, losses.avg


def validate(val_loader, model, criterion, opt):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):
            input = input.float()
            if opt.gpu is not None:
                input = input.cuda(opt.gpu, non_blocking=True)
            input = input.float().cuda()
            target = target.cuda(opt.gpu, non_blocking=True)

            # compute output
            feat = model(input).cuda()
            feat = feat.detach()
            output = feat
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top3.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top3.val:.3f} ({top3.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top3=top3))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top3.avg:.3f}'
              .format(top1=top1, top3=top3))

    return top1.avg, top3.avg, losses.avg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    best_acc1 = 0
    main()
