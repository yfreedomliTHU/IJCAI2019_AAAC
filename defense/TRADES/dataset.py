import os
import glob

import torch

import PIL
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple





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
                 filenames1: str,
                 filenames2: str,
                 mode : str,
                 transform = None):

        self.filenames1 = filenames1
        self.filenames2 = filenames2
        self.mode = mode
        self.transform = transform


    def __len__(self):
        return len(self.filenames1)

    def __getitem__(self, index: int):
        """ Returns one item from the dataset

        Parameters
        ==========
        index : int
            The index of the item

        Returns:
        tuple (image, label):
            Label is index of the label class. label is -1 if test mode
        """
        img_name1 = self.filenames1[index]
        img_name2 = self.filenames2[index]
        if self.mode == 'train':
            label1 = int(img_name1.split('/')[-2])
            label2 = int(img_name2.split('/')[-2])
        elif self.mode == 'test':
            label1 = -1
            label2 = -1
        else:
            raise ValueError('Warning:the mode should be train or test!')

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img1 = Image.open(img_name1).convert('RGB')
        img2 = Image.open(img_name2).convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, label1, img2, label2



if __name__ == '__main__':
    # have your data stored in the DATA folder
    from torchvision import transforms
    from random import randint
    #example to use ACCC_dataset

    input_path = r'your data_path'
    img_png = glob.glob(os.path.join(input_path, "./*/*.png"))
    img_jpg = glob.glob(os.path.join(input_path, "./*/*.jpg"))
    filenames = img_png + img_jpg
    #print(filenames)
    train_size = int(len(filenames) * 0.8)
    test_size = len(filenames) - train_size
    train_filenames = filenames[0:train_size]
    test_filenames = filenames[train_size:]
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

