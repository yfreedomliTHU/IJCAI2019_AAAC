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
from torch.utils.data import Dataset
import torchvision.transforms as transforms


from cnn_finetune import make_model
import PIL
from PIL import Image
from typing import Tuple

begin_time = time.time()

def parse_args():

    parser = argparse.ArgumentParser(description='IJCAI_2019_AAAC_defense')

    parser.add_argument('--input_dir', metavar='DIR', default='',
                        help='Input directory with images.')
    parser.add_argument('--output_file', metavar='FILE', default='',
                        help='Output file to save labels.')
    parser.add_argument('--arch', '-a', default='polynet',
                        help='model architecture: ')
    parser.add_argument('--checkpoint_inception_v4', default='./model/checkpoint_inception_v4.pth',
                        help='Path to model checkpoint.')
    parser.add_argument('--checkpoint_inception_v3', default='./model/checkpoint_inception_v3.pth',
                        help='Path to model checkpoint.')
    parser.add_argument('--checkpoint_polynet', default='./model/checkpoint_polynet.pth',
                        help='Path to model checkpoint.')
    parser.add_argument('--image_size', type=int, default=224, metavar='N',
                        help='Image patch size (default: 224)')
    parser.add_argument('--batch-size', type=int, default=22, metavar='N',
                        help='Batch size')
    parser.add_argument('--gpu_id', default=0, nargs='+', help='gpu ids to use, e.g. 0 1 2 3', type=int)
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')

    return parser.parse_args()

class AAAC_dataset(Dataset):
    """IJCAI_AAAC Dataset loader

    Parameters
    ==========
    filenames : string
        Root directory of dataset
    mode: string
        train or test
    transform : callable, optional
        A function/transform that  takes in an PIL image and returns a
        transformed version. E.g, ``transforms.RandomCrop``
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    """

    def __init__(self,
                 filenames: str,
                 mode : str,
                 transform = None):

        self.filenames = filenames
        self.mode = mode
        self.transform = transform


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        """ Returns one item from the dataset

        Parameters
        ==========
        index : int
            The index of the item

        Returns:
        tuple (image, label):
            Label is index of the label class. label is -1 if test mode
        """
        img_name = self.filenames[index]
        if self.mode == 'train':
            label = int(img_name.split('/')[-2])
        elif self.mode == 'test':
            label = -1
        else:
            raise ValueError('Warning:the mode should be train or test!')

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

class NormalizedModel(nn.Module):
    """
    Wrapper for a model to account for the mean and std of a dataset.
    mean and std do not require grad as they should not be learned, but determined beforehand.
    mean and std should be broadcastable (see pytorch doc on broadcasting) with the data.
    Args:
        model (nn.Module): model used to predict
        mean (torch.Tensor): sequence of means for each channel
        std (torch.Tensor): sequence of standard deviations for each channel
    """

    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor) -> None:
        super(NormalizedModel, self).__init__()

        self.model = model
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)
        #print(self.mean.device)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #print('before:',self.mean.device)
        normalized_input = (input - self.mean) / self.std

        return self.model(normalized_input)

def main():

    global args
    args = parse_args()
    print(args)
    gpu_id = args.gpu_id
    if isinstance(gpu_id, int):
        gpu_id = [gpu_id]
    print("Let's use ", len(gpu_id), " GPUs!")
    print("gpu_ids:", gpu_id)

    if not os.path.exists(args.input_dir):
        print("Error: Invalid input folder %s" % args.input_dir)
        exit(-1)
    if not args.output_file:
        print("Error: Please specify an output file")
        exit(-1)
    device = torch.device('cuda: %d' % gpu_id[0] if torch.cuda.is_available() else 'cpu')
    if 'inception' in args.arch.lower():
        print('Using Inception Normalization!')
        image_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    else:
        print('Using Imagenet Normalization!')
        image_mean = torch.tensor([0.4802, 0.4481, 0.3975]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.2770, 0.2691, 0.2821]).view(1, 3, 1, 1)
    '''
    model_init_incpv4 = make_model('inception_v4', 110, pretrained=False)
    model_incepv4 = NormalizedModel(model_init_incpv4, image_mean, image_std)
    model_incepv4.to(device)

    model_init_incpv3 = make_model('inception_v3', 110, pretrained=False)
    model_incepv3 = NormalizedModel(model_init_incpv3, image_mean, image_std)
    model_incepv3.to(device)
    
    model_init_inresv2 = make_model('inceptionresnetv2', 110, pretrained=False)
    model_inresv2 = NormalizedModel(model_init_inresv2, image_mean, image_std)
    model_inresv2.to(device)
    '''
    model_init_pn = make_model('polynet', 110, pretrained=False)
    model_pn = NormalizedModel(model_init_pn, image_mean, image_std)
    model_pn.to(device)


    # Load Data
    input_path = args.input_dir
    img_png = glob.glob(os.path.join(input_path, "*.png"))
    img_jpg = glob.glob(os.path.join(input_path, "*.jpg"))
    filenames = img_png + img_jpg


    test_transform = transforms.Compose([
        transforms.Resize([args.image_size, args.image_size], interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor()
    ])
    test_dataset = AAAC_dataset(filenames, mode='test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    #Load checkpoint
    '''
    checkpoint1 = torch.load(args.checkpoint_inception_v4, map_location=device)
    model_incepv4.load_state_dict(checkpoint1['state_dict'], strict=True)
    model_incepv4.eval()

    checkpoint2 = torch.load(args.checkpoint_inception_v3, map_location=device)
    model_incepv3.load_state_dict(checkpoint2['state_dict'], strict=True)
    model_incepv3.eval()
    '''
    checkpoint3 = torch.load(args.checkpoint_polynet, map_location=device)

    # multi_gpu training
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint3['state_dict'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model_pn.load_state_dict(new_state_dict, strict=True)

    #model_inresv2.load_state_dict(checkpoint3['state_dict'], strict=True)
    model_pn.eval()

    cudnn.benchmark = False

    '''
    model_test = {
        "incepv4":model_incepv4,
        "incepv3" :model_incepv3,
        "inresv2" :model_inresv2,
    }
    pre_out = {
        "incepv4": [],
        "incepv3" : [],
        "inresv2" : [],
    }
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(device)
            for key, model_pre in model_test.items():
                logits = model_pre(data)
                pre_out[key].append(logits.cpu().numpy())

    for key, pre in pre_out.items():
        pre = np.concatenate(np.array(pre))
        pre_out[key] = pre
    #print(pre_out['inresv2'],pre_out['incepv3'],pre_out['resnet50'])
    ans_sum = 3 * pre_out['inresv2'] + 2 * pre_out['incepv4'] + pre_out['incepv3']
    outputs = np.argmax(ans_sum, axis=-1)
    '''
    #single model:
    pre_out = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(device)
            logits = model_pn(data)
            pre_out.append(logits.cpu().numpy())

    pre_out = np.concatenate(np.array(pre_out))
    outputs = np.argmax(pre_out, axis=-1)
    

    print(outputs)

    with open(args.output_file, 'w') as out_file:
        filenames = filenames
        for filename, label in zip(filenames, outputs):
            filename = os.path.basename(filename)
            out_file.write('{0},{1}\n'.format(filename, label))

    print('Time cost is %f s' % (time.time() - begin_time))


if __name__ == '__main__':
    main()