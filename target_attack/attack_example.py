# Attack code
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from attack import attack
from scipy.misc import imread, imresize
import numpy as np
import time
from typing import Tuple, Optional
import torch.optim as optim
from cnn_finetune import make_model
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL
from PIL import Image

begin_time = time.time()

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
                 input_path,
                 filename,
                 label,
                 transform = None):

        self.input_path = input_path
        self.filename = filename
        self.label = label
        self.transform = transform


    def __len__(self):
        return len(self.filename)

    def __getitem__(self, index: int):

        img_name = self.filename[index]
        label = self.label[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img_path = os.path.join(self.input_path, img_name)
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label,img_name

class FGM_L2:
    """
    Fast Gradient Method using L2 distance.
    Parameters
    ==========
    eps : float
        Epsilon to multiply the attack noise
    image_constraints : tuple
        Bounds of the images. Default: (0, 1)
    """

    def __init__(self,
                 eps: float,
                 image_constraints: Tuple[float, float] = (0, 1)) -> None:
        self.eps = eps

        self.boxmin = image_constraints[0]
        self.boxmax = image_constraints[1]

        self.criterion = F.cross_entropy

    def attack(self, model: nn.Module, inputs: torch.Tensor,
               labels: torch.Tensor, targeted: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the given inputs.
        Parameters
        ==========
        model : nn.Module
            Model to attack
        inputs : torch.Tensor
            Batch of images to generate adv for
        labels : torch.Tensor
            True labels in case of untargeted, target in case of targeted
        targeted : bool
            Whether to perform a targeted attack or not
        """
        multiplier = -1 if targeted else 1
        delta = torch.zeros_like(inputs, requires_grad=True)

        logits = model(inputs + delta)
        loss = self.criterion(logits, labels)
        grad = torch.autograd.grad(loss, delta)[0]

        adv = inputs + multiplier * self.eps * grad / (grad.view(grad.size(0), -1).norm(2, 1)).view(-1, 1, 1, 1)
        adv = torch.clamp(adv, self.boxmin, self.boxmax)

        return adv.detach()

class DDN:
    """
    DDN attack: decoupling the direction and norm of the perturbation to achieve a small L2 norm in few steps.

    Parameters
    ----------
    steps : int
        Number of steps for the optimization.
    gamma : float, optional
        Factor by which the norm will be modified. new_norm = norm * (1 + or - gamma).
    init_norm : float, optional
        Initial value for the norm.
    quantize : bool, optional
        If True, the returned adversarials will have quantized values to the specified number of levels.
    levels : int, optional
        Number of levels to use for quantization (e.g. 256 for 8 bit images).
    max_norm : float or None, optional
        If specified, the norms of the perturbations will not be greater than this value which might lower success rate.
    device : torch.device, optional
        Device on which to perform the attack.
    callback : object, optional
        Visdom callback to display various metrics.

    """

    def __init__(self,
                 steps: int,
                 gamma: float = 0.05,
                 init_norm: float = 1.,
                 quantize: bool = True,
                 levels: int = 256,
                 max_norm: Optional[float] = None,
                 device: torch.device = torch.device('cpu'),
                 callback: Optional = None) -> None:
        self.steps = steps
        self.gamma = gamma
        self.init_norm = init_norm

        self.quantize = quantize
        self.levels = levels
        self.max_norm = max_norm

        self.device = device
        self.callback = callback

    def attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
               targeted: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.

        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : torch.Tensor
            Labels of the samples to attack if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.

        Returns
        -------
        torch.Tensor
            Batch of samples modified to be adversarial to the model.

        """
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
        batch_size = inputs.shape[0]
        multiplier = 1 if targeted else -1
        delta = torch.zeros_like(inputs, requires_grad=True)
        norm = torch.full((batch_size,), self.init_norm, device=self.device, dtype=torch.float)
        worst_norm = torch.max(inputs, 1 - inputs).view(batch_size, -1).norm(p=2, dim=1)

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.steps, eta_min=0.01)

        best_l2 = worst_norm.clone()
        best_delta = torch.zeros_like(inputs)
        adv_found = torch.zeros(inputs.size(0), dtype=torch.uint8, device=self.device)

        for i in range(self.steps):
            scheduler.step()

            l2 = delta.data.view(batch_size, -1).norm(p=2, dim=1)
            adv = inputs + delta
            logits = model(adv)
            pred_labels = logits.argmax(1)
            ce_loss = F.cross_entropy(logits, labels, reduction='sum')
            loss = multiplier * ce_loss

            is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
            is_smaller = l2 < best_l2
            is_both = is_adv * is_smaller
            adv_found[is_both] = 1
            best_l2[is_both] = l2[is_both]
            best_delta[is_both] = delta.data[is_both]

            optimizer.zero_grad()
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            if self.callback:
                cosine = F.cosine_similarity(-delta.grad.view(batch_size, -1),
                                             delta.data.view(batch_size, -1), dim=1).mean().item()
                self.callback.scalar('ce', i, ce_loss.item() / batch_size)
                self.callback.scalars(
                    ['max_norm', 'l2', 'best_l2'], i,
                    [norm.mean().item(), l2.mean().item(),
                     best_l2[adv_found].mean().item() if adv_found.any() else norm.mean().item()]
                )
                self.callback.scalars(['cosine', 'lr', 'success'], i,
                                      [cosine, optimizer.param_groups[0]['lr'], adv_found.float().mean().item()])

            optimizer.step()

            norm.mul_(1 - (2 * is_adv.float() - 1) * self.gamma)
            norm = torch.min(norm, worst_norm)

            delta.data.mul_((norm / delta.data.view(batch_size, -1).norm(2, 1)).view(-1, 1, 1, 1))
            delta.data.add_(inputs)
            if self.quantize:
                delta.data.mul_(self.levels - 1).round_().div_(self.levels - 1)
            delta.data.clamp_(0, 1).sub_(inputs)

        if self.max_norm:
            best_delta.renorm_(p=2, dim=0, maxnorm=self.max_norm)
            if self.quantize:
                best_delta.mul_(self.levels - 1).round_().div_(self.levels - 1)

        return inputs + best_delta

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

def parse_args():

    parser = argparse.ArgumentParser(description='AAAC_Attack ')
    parser.add_argument('--input_dir', default='', type=str,
                        help='Input directory with images.')
    parser.add_argument('--output_dir', default='', type=str,
                        help='Output directrory with adv_images.')
    parser.add_argument('--model-path', '--m', default='./model/checkpoint_se_resnext50_32x4d.pth',
                        help='model path.')
    parser.add_argument('--sm2', default='./model/checkpoint_inceptionresnetv2.pth',
                        help='surrogate-model-path')
    parser.add_argument('--image_size', type=int, default=224, metavar='N',
                        help='Image patch size (default: 224)')
    parser.add_argument('--batch_size', type=int, default=33, metavar='N',
                        help='Batch size')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--gpu_id', default=0, nargs='+',help='gpu ids to use, e.g. 0 1 2 3', type=int)

    return parser.parse_args()


def generate_adv(test_loader, device, model, surrogate_models):

    # Load image
    # t_img = torch.from_numpy(img).float().div(255).permute(2, 0, 1).unsqueeze(0)



    # Sanity check: image correctly labeled:
    # t_img = t_img.to(device)
    #assert model(t_img).argmax() == label
    #assert black_box_model(img) == label



    # Sanity check: image correctly labeled by surrogate classifier:
    # assert smodel(t_img).argmax() == label


    attacks = [
        DDN(100, device=device),
        FGM_L2(1)
    ]
    #tensor in cpu
    for i, (data, labels,filename) in enumerate(test_loader):
        adv = attack(model, surrogate_models, attacks,
                     data, labels, targeted=True, device=device,)
        adv = np.array(adv)
    #pred_on_adv = black_box_model(adv)
    #print('True label: {}; Prediction on the adversarial: {}'.format(label,
    #                                                                 pred_on_adv))

    # Compute l2 norm in range [0, 1]
        for mm in range(len(data)):
            img = data[mm].permute(1, 2, 0).cpu().numpy() * 255
            l2_norm = np.linalg.norm(((adv[mm] - img) / 255))
            print('L2 norm of the attack: {:.4f}'.format(l2_norm))
            adv_save = imresize(adv[mm], [299, 299])
            Image.fromarray(adv_save).save(os.path.join(args.output_dir, filename[mm]), format='PNG')



def main():
    global args
    args = parse_args()
    print(args)

    gpu_id = args.gpu_id
    if isinstance(gpu_id, int):
        gpu_id = [gpu_id]

    print("Let's use ", len(gpu_id), " GPUs!")
    print("gpu_ids:", gpu_id)

    #Normalization:inception and imagenet
    image_mean_inc = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    image_std_inc = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    image_mean_img = torch.tensor([0.4802, 0.4481, 0.3975]).view(1, 3, 1, 1)
    image_std_img = torch.tensor([0.2770, 0.2691, 0.2821]).view(1, 3, 1, 1)


    device = torch.device('cuda: %d' %gpu_id[0]  if torch.cuda.is_available() else 'cpu')


    # Load model under attack:resnet50
    m = make_model('se_resnext50_32x4d', 110, pretrained=False)
    model = NormalizedModel(m, image_mean_img, image_std_img)
    model.to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict['state_dict'], strict=True)
    model.eval()




    # Load surrogate model
    '''
    smodel1 = make_model('inception_v3', 110, pretrained=False)
    smodel1 = NormalizedModel(smodel1, image_mean_inc, image_std_inc)
    smodel1.to(device)
    state_dict_s1 = torch.load(args.sm1, map_location=device)
    smodel1.load_state_dict(state_dict_s1['state_dict'], strict=True)
    smodel1.eval()
    '''

    smodel2 = make_model('inceptionresnetv2', 110, pretrained=False)
    smodel2 = NormalizedModel(smodel2, image_mean_inc, image_std_inc)
    smodel2.to(device)
    state_dict_s2 = torch.load(args.sm2, map_location=device)
    smodel2.load_state_dict(state_dict_s2['state_dict'], strict=True)
    smodel2.eval()

    surrogate_models = [smodel2]

    test_transform = transforms.Compose([
        transforms.Resize([args.image_size, args.image_size], interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor()
    ])

    dev = pd.read_csv(os.path.join(args.input_dir, 'dev.csv'))
    #target:filename2label = {dev.iloc[i]['filename']: dev.iloc[i]['targetedLabel'] for i in range(len(dev))}
    #filename2label = {dev.iloc[i]['filename']: dev.iloc[i]['trueLabel'] for i in range(len(dev))}
    filename = []
    label = []
    for i in range(len(dev)):
        filename.append(dev.iloc[i]['filename'])
        label.append(int(dev.iloc[i]['targetedLabel']))

    test_dataset = AAAC_dataset(args.input_dir, filename, label, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers)

    generate_adv(test_loader, device, model, surrogate_models)


    '''
    for filename in filename2label.keys():
        img = imread(os.path.join(args.input_dir, filename), mode='RGB')
        img = imresize(img, [args.image_size, args.image_size])
        label = filename2label[filename]

        adv = generate_adv(img, label, device, model, smodel)

        adv = imresize(adv, [299, 299])
        Image.fromarray(adv).save(os.path.join(args.output_dir, filename), format='PNG')
    '''

    print('Time cost is %f s' % (time.time() - begin_time))


if __name__ == '__main__':
    main()
