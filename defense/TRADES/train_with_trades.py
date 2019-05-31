from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import glob
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import PIL
from PIL import Image

import dataset
import utils
from trades import trades_loss

from fastai.vision import *
from cnn_finetune import make_model

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3, 4, 5"


def parse_args():

    parser = argparse.ArgumentParser(description='AAAC Adversarial Training with trades loss')

    parser.add_argument('--trades', action='store_true', help='adversarial training with TRADES')

    parser.add_argument('--data_path', default=r'../IJCAI_2019_AAAC_train_processed',
                        type=str, help='path to dataset')
    parser.add_argument('--data_path_adv1', default=r'../IJCAI_2019_AAAC_train_MIFSGM',
                        type=str, help='path to dataset')
    parser.add_argument('--data_path_adv2', default=r'../IJCAI_2019_AAAC_train_PGD_inception_v1',
                        type=str, help='path to dataset')
    parser.add_argument('--arch', '-a', default='inceptionresnetv2',
                        help='model architecture: ' )
    parser.add_argument('--save-folder', '--sf', default='AAAC_trades', required=True, type=str, help='folder where the models will be saved')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')

    parser.add_argument('--evaluate', '--eval', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--pretrained', '--pt', dest='pretrained', action='store_true', help='use pre-trained model')

    parser.add_argument('--batch-size', '-b', default=128, type=int, help='mini-batch size')
    parser.add_argument('--epochs', '-e', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--lr-step', '--learning-rate-step', default=5, type=int,
                        help='step size for learning rate decrease')
    parser.add_argument('--momentum', default=0.9, type=float, help=' SDG momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay')


    parser.add_argument('--print-freq', '--pf', default=10, type=int, help='print frequency')
    parser.add_argument('--image_size', '--img_size', default=299, type=int, help='set image_size for model.')
    parser.add_argument('--gpu_id', default=4, nargs='+',help='gpu ids to use, e.g. 0 1 2 3', type=int)
    parser.add_argument('--epsilon', default=0.3,
                        help='perturbation')
    parser.add_argument('--num-steps', default=40,
                        help='perturb number of steps')
    parser.add_argument('--step-size', default=0.01,
                        help='perturb step size')
    parser.add_argument('--beta', default=1.0,
                        help='regularization, i.e., 1/lambda in TRADES')
    parser.add_argument('--distance', default='l_2', type=str,
                        help='distance type, i.e., l_2 or l_inf')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-freq', '-s', default=5, type=int, metavar='N',
                        help='save frequency')

    return parser.parse_args()


def train(args, model, device, train_loader, optimizer, epoch):
    cudnn.benchmark = True
    model.train()
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    end = time.time()
    for batch_idx, (data1, target1, data2, target2) in enumerate(train_loader):
        if batch_idx % 2 == 0:
            data, target = data1.to(device), target1.to(device)
        else:
            data, target = data2.to(device), target2.to(device)


        optimizer.zero_grad()

        if args.trades:
            # calculate robust loss
            loss = trades_loss(model=model,
                               x_natural=data,
                               y=target,
                               optimizer=optimizer,
                               step_size=args.step_size,
                               epsilon=args.epsilon,
                               perturb_steps=args.num_steps,
                               beta=args.beta,
                               distance=args.distance,
                               device=device)
        else:
            # CE loss
            loss = F.cross_entropy(model(data), target)

        loss.backward()
        optimizer.step()

        batch_time.append(time.time() - end)
        losses.append(loss.item())
        end = time.time()

        # print progress
        if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
            print('Epoch: {} [{}/{} ({:.3f}%)] Time {batch_time.last_avg:.3f}'
                  '\tLoss: {loss.last_avg:.4f}'.format(epoch, batch_idx + 1, len(train_loader), (batch_idx + 1) / len(train_loader),
                                                       batch_time=batch_time, loss=losses))
def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data1, target1, data2, target2) in enumerate(train_loader):
            if batch_idx % 2 == 0:
                data, target = data1.to(device), target1.to(device)
            else:
                data, target = data2.to(device), target2.to(device)

            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data1, target1, data2, target2) in enumerate(test_loader):
            if batch_idx % 2 == 0:
                data, target = data1.to(device), target1.to(device)
            else:
                data, target = data2.to(device), target2.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def get_adjusted_lr(args, optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch <= int(args.epochs * 0.5):
        lr = args.lr
    elif epoch <= int(args.epochs * 0.9):
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
# Main
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
    # set gpu id
    device = torch.device('cuda: %d' %gpu_id[0]  if torch.cuda.is_available() else 'cpu')
    # set random seed
    #torch.manual_seed(args.seed)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))
    model = make_model(args.arch, 110, pretrained=args.pretrained)

    # model information
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

    # Load Data
    clean_input_path = args.data_path
    adv_input_path1 = args.data_path_adv1
    adv_input_path2 = args.data_path_adv2
    img_clean = glob.glob(os.path.join(clean_input_path, "./*/*.jpg"))
    img_adv1 = glob.glob(os.path.join(adv_input_path1, "./*/*.jpg"))
    img_adv2 = glob.glob(os.path.join(adv_input_path2, "./*/*.jpg"))
    random.shuffle(img_clean)
    random.shuffle(img_adv1)
    random.shuffle(img_adv2)
    train_size_clean = int(len(img_clean) * 0.9)
    train_size_adv = int(len(img_adv1) * 0.9)

    train_filenames1 = img_clean[0:train_size_clean] + img_adv1[0:train_size_adv]
    train_filenames2 = img_clean[0:train_size_clean] + img_adv1[0:train_size_adv]
    random.shuffle(train_filenames1)
    random.shuffle(train_filenames2)
    test_filenames1 = img_clean[train_size_clean:] + img_adv1[train_size_adv:]
    test_filenames2 = img_clean[train_size_clean:] + img_adv2[train_size_adv:]
    random.shuffle(test_filenames1)
    random.shuffle(test_filenames2)
    '''
    train_size = int(len(filenames1) * 0.8)
    test_size = len(filenames1) - train_size
    train_filenames = filenames[0:train_size]
    test_filenames = filenames[train_size:]
    '''

    if 'inception' in args.arch.lower():
        print('Using Inception Normalization!')
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize([args.image_size, args.image_size], interpolation=PIL.Image.BILINEAR),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        test_transform = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size], interpolation=PIL.Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
    else:
        print('Using Imagenet Normalization!')
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize([args.image_size, args.image_size], interpolation=PIL.Image.BILINEAR),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size], interpolation=PIL.Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
        ])

    train_dataset = dataset.AAAC_dataset(train_filenames1, train_filenames2, mode='train', transform=train_transform)
    test_dataset = dataset.AAAC_dataset(test_filenames1,test_filenames2, mode='train', transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                             pin_memory=True)

    # Train
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        get_adjusted_lr(args, optimizer, epoch)

        # adversarial training
        print('Learning rate for epoch {}: {:.2e}'.format(epoch, optimizer.param_groups[0]['lr']))
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        print('================================================================')
        eval_train(model, device, train_loader)
        _, val_acc = eval_test(model, device, test_loader)
        print('================================================================')

        # save checkpoint
        utils.save_checkpoint(
            state={'epoch': epoch + 1,
                   'arch': args.arch,
                   'state_dict': model.state_dict(),
                   'val_acc': val_acc,
                   'optimizer': optimizer.state_dict()},
            filename=os.path.join(args.save_folder, 'checkpoint_{}.pth'.format(args.arch)))

        utils.save_checkpoint(
            state=model.state_dict(),
            filename=os.path.join(args.save_folder, '{}_epoch-{}.pt'.format(args.arch, epoch + 1)),
            cpu=True
        )


if __name__ == '__main__':
    main()





