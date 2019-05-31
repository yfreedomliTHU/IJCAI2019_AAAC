import os
import glob
import numpy as np
import torch

import PIL
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple
#from fastai.vision import *




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



if __name__ == '__main__':
    # have your data stored in the DATA folder
    from torchvision import transforms
    from random import randint
    #example to use ACCC_dataset
    # test the dataset
    input_path = r'your data path'
    img_png = glob.glob(os.path.join(input_path, "./*/*.png"))
    img_jpg = glob.glob(os.path.join(input_path, "./*/*.jpg"))
    filenames = img_png + img_jpg
    #print(filenames)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize([224, 224], interpolation=PIL.Image.BILINEAR),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor()
    ])
    '''
    tfms_imagenet = transforms.Compose([
        transforms.Resize([299, 299]),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_stats[0], imagenet_stats[1])
    ])
    '''
    tfms_incep = transforms.Compose([
        transforms.Resize([299, 299]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    #transform = transforms.Compose([transforms.ToTensor()])
    dataset = AAAC_dataset(filenames, mode='train', transform=tfms_incep)
    print('dataset has', len(dataset), 'samples')
    tensor, label = dataset[randint(0, len(dataset) - 1)]
    print('Sample shape:', tensor)
    print('label:', label)
    #you can get train_data and val_data via [train_size,val_size]
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

    print('train_data has', len(train_data), 'samples')



