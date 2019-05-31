import argparse
import os
import time
import glob
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import PIL
from PIL import Image

import models
import dataset
import utils
from fast_adv.attacks import DDN

from fastai.vision import *
from cnn_finetune import make_model


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 4, 5"

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def parse_args():

    parser = argparse.ArgumentParser(description='Adversarial AAAC Training with the DDN Attack')

    parser.add_argument('--data_path', default=r'../IJCAI_2019_AAAC_train_data',
                        type=str, help='path to dataset')
    parser.add_argument('--arch', '-a', default='resnet50',
                        help='model architecture: ' )
    parser.add_argument('--save-folder', '--sf', default='AAAC_ddn', required=True, type=str, help='folder where the models will be saved')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')

    parser.add_argument('--evaluate', '--eval', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', type=str, help='path to latest checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--pretrained', '--pt', dest='pretrained', action='store_true', help='use pre-trained model')

    parser.add_argument('--batch-size', '-b', default=4, type=int, help='mini-batch size')
    parser.add_argument('--epochs', '-e', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.005, type=float, help='initial learning rate')
    parser.add_argument('--lr-step', '--learning-rate-step', default=5, type=int,
                        help='step size for learning rate decrease')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay')

    parser.add_argument('--adv', action='store_true', help='Use adversarial training')
    parser.add_argument('--start-adv-epoch', '--sae', type=int, default=0,
                        help='epoch to start training with adversarial images')
    parser.add_argument('--max-norm', default=1, type=float, help='max norm for the adversarial perturbations')
    parser.add_argument('--steps', default=100, type=int, help='number of steps for the attack')

    parser.add_argument('--visdom-port', '--vp', type=int, help='For visualization, which port visdom is running.')
    parser.add_argument('--print-freq', '--pf', default=10, type=int, help='print frequency')
    parser.add_argument('--image_size', '--img_size', default=224, type=int, help='set image_size for model.')
    parser.add_argument('--gpu_id', default=4, nargs='+',help='gpu ids to use, e.g. 0 1 2 3', type=int)

    return parser.parse_args()

#About Training

def main():
    global args

    args = parse_args()
    print(args)
    gpu_id = args.gpu_id
    if isinstance(gpu_id, int):
        gpu_id = [gpu_id]

    print("Let's use ", len(gpu_id), " GPUs!")
    print("gpu_ids:", gpu_id)
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    device = torch.device('cuda: %d' %gpu_id[0]  if torch.cuda.is_available() else 'cpu')
    #device = gpu_id
    if 'inception' in args.arch.lower():
        print('Using Inception Normalization!')
        image_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    else:
        print('Using Imagenet Normalization!')
        image_mean = torch.tensor([0.4802, 0.4481, 0.3975]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.2770, 0.2691, 0.2821]).view(1, 3, 1, 1)

    # create model
    '''
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        m = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        m = models.__dict__[args.arch]()
    '''
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))
    model_init = make_model(args.arch, 110, pretrained=args.pretrained)
    #model = utils.NormalizedModel(m, image_mean, image_std)
    model = utils.NormalizedModel(model_init, image_mean, image_std)
    print('model_struct:', model)
    print('model parameters:', sum(param.numel() for param in model.parameters()))
    '''
    for param in model.parameters():
        print(param.shape)
        print(param.numel())
    '''
    model.to(device)


    if len(gpu_id) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_id)



    '''
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    '''


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            prec1 = checkpoint['prec1']
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict,strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.last_epoch = checkpoint['epoch'] - 1
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    input_path = args.data_path
    clean_path = os.path.join(input_path, 'IJCAI_2019_AAAC_train')
    adv_path = os.path.join(input_path, 'IJCAI_2019_AAAC_train_adv')
    img_clean = glob.glob(os.path.join(clean_path, "./*/*.jpg"))
    img_adv = glob.glob(os.path.join(adv_path, "./*/*.jpg"))
    filenames = img_clean + img_adv

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize([args.image_size, args.image_size], interpolation=PIL.Image.BILINEAR),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize([args.image_size, args.image_size], interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor()
    ])

    all_dataset = dataset.AAAC_dataset(filenames, mode='train', transform=train_transform)
    #val_dataset = dataset.TinyImageNet(args.data, mode='val', transform=test_transform)
    train_size = int(0.8 * len(all_dataset))
    val_size = len(all_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(all_dataset, [train_size, val_size])

    '''
    if args.visdom_port:
        from visdom_logger.logger import VisdomLogger
        callback = VisdomLogger(port=args.visdom_port)
    else:
        callback = None
    '''
    callback = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                               num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                             pin_memory=True)

    attacker = DDN(steps=args.steps, device=device)

    if args.evaluate:
        validate(val_loader, model, criterion, device, 0, callback=callback)
        return

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        print('Learning rate for epoch {}: {:.2e}'.format(epoch, optimizer.param_groups[0]['lr']))

        # train for one epoch
        train(train_loader, model, model_init, criterion, optimizer, attacker, device, epoch, callback)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, device, epoch + 1, callback)

        utils.save_checkpoint(
            state={'epoch': epoch + 1,
                   'arch': args.arch,
                   'state_dict': model.state_dict(),
                   'prec1': prec1,
                   'optimizer': optimizer.state_dict()},
            filename=os.path.join(args.save_folder, 'checkpoint_{}.pth'.format(args.arch)))

        utils.save_checkpoint(
            state=model.state_dict(),
            filename=os.path.join(args.save_folder, '{}_epoch-{}.pt'.format(args.arch, epoch + 1)),
            cpu=True
        )


def train(train_loader, model, model_init, criterion, optimizer, attacker, device, epoch, callback=None):

    cudnn.benchmark = True
    model.train()
    length = len(train_loader)

    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    losses_adv = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    adv_acc = utils.AverageMeter()
    l2_adv = utils.AverageMeter()

    end = time.time()
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device, non_blocking=True)

        if args.adv and epoch >= args.start_adv_epoch:
            '''
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            '''
            model.eval()
            '''
            if len(gpu_id) > 1:
                model = torch.nn.DataParallel(model, device_ids=gpu_id)
            '''

            utils.requires_grad_(model_init, False)
            with torch.no_grad():
                clean_logits = model(data)
            loss = criterion(clean_logits, labels)

            adv = attacker.attack(model, data, labels)

            l2_norms = (adv - data).view(args.batch_size, -1).norm(2, 1)
            mean_norm = l2_norms.mean()
            if args.max_norm:
                adv = torch.renorm(adv - data, p=2, dim=0, maxnorm=args.max_norm) + data
            l2_adv.append(mean_norm.item())

            utils.requires_grad_(model_init, True)




            '''
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            '''
            model.train()
            '''
            if len(gpu_id) > 1:
                model = torch.nn.DataParallel(model, device_ids=gpu_id)
            '''
            adv_logits = model(adv.detach())
            loss_adv = criterion(adv_logits, labels)

            loss_to_optimize = loss_adv

            losses_adv.append(loss_adv.item())
            l2_adv.append((adv - data).view(args.batch_size, -1).norm(p=2, dim=1).mean().item())
            adv_acc.append((adv_logits.argmax(1) == labels).sum().item() / args.batch_size)
        else:
            clean_logits = model(data)
            loss = criterion(clean_logits, labels)
            loss_to_optimize = loss

        optimizer.zero_grad()
        loss_to_optimize.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(clean_logits, labels, topk=(1, 5))
        losses.append(loss.item())
        top1.append(prec1)
        top5.append(prec5)

        # measure elapsed time
        batch_time.append(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0 or (i + 1) == length:

            if args.adv and epoch >= args.start_adv_epoch:
                print('Epoch: [{0:>2d}][{1:>3d}/{2:>3d}] Time {batch_time.last_avg:.3f}'
                      '\tLoss {loss.last_avg:.4f}\tAdv {loss_adv.last_avg:.4f}'
                      '\tPrec@1 {top1.last_avg:.3%}\tPrec@5 {top5.last_avg:.3%}'.format(epoch, i + 1, len(train_loader),
                                                                                        batch_time=batch_time,
                                                                                        loss=losses,
                                                                                        loss_adv=losses_adv,
                                                                                        top1=top1, top5=top5))
            else:
                print('Epoch: [{0:>2d}][{1:>3d}/{2:>3d}] Time {batch_time.last_avg:.3f}\tLoss {loss.last_avg:.4f}'
                      '\tPrec@1 {top1.last_avg:.3%}\tPrec@5 {top5.last_avg:.3%}'.format(epoch, i + 1, len(train_loader),
                                                                                        batch_time=batch_time,
                                                                                        loss=losses,
                                                                                        top1=top1, top5=top5))

            if callback:
                if args.adv and epoch >= args.start_adv_epoch:
                    callback.scalars(['train_loss', 'adv_loss'], i / length + epoch,
                                     [losses.last_avg, losses_adv.last_avg])
                    callback.scalars(['train_prec@1', 'train_prec@5', 'adv_acc'], i / length + epoch,
                                     [top1.last_avg * 100, top5.last_avg * 100, adv_acc.last_avg * 100])
                    callback.scalar('adv_l2', i / length + epoch, l2_adv.last_avg)

                else:
                    callback.scalar('train_loss', i / length + epoch, losses.last_avg)
                    callback.scalars(['train_prec@1', 'train_prec@5'], i / length + epoch,
                                     [top1.last_avg * 100, top5.last_avg * 100])


def validate(val_loader, model, criterion, device, epoch, callback=None):
    model.eval()
    cudnn.benchmark = False

    batch_time = utils.AverageMeter()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        end = time.time()
        for i, (data, labels) in enumerate(val_loader):
            labels = labels.to(device, non_blocking=True)
            data = data.to(device)

            # compute output
            output = model(data)
            #loss = criterion(output, labels)
            all_logits.append(output)
            all_labels.append(labels)

            batch_time.append(time.time() - end)
            end = time.time()

        all_logits = torch.cat(all_logits, 0)
        all_labels = torch.cat(all_labels, 0)
        # measure accuracy and record loss for clean samples
        loss = criterion(output, labels).item()
        prec1, prec5 = utils.accuracy(all_logits, all_labels, topk=(1, 5))

    print('Val | Time {:.3f}\tLoss {:.4f} | Clean: Prec@1 {:.3%}\tPrec@5 {:.3%}'.format(batch_time.avg, loss,
                                                                                        prec1, prec5))
    if callback:
        callback.scalar('val_loss', epoch, loss)
        callback.scalars(['val_prec@1', 'val_prec@5'], epoch, [prec1, prec5])

    return prec1


if __name__ == '__main__':
    main()

    '''
    if len(gpu_id) > 1:
        t.save(net.module.state_dict(), "model.pth")
    else:
        t.save(net.state_dict(), "model.pth")
    '''