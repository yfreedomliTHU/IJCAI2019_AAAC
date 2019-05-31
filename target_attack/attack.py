import torch
import numpy as np
from typing import Iterable, Optional


# Simulate a black-box model (i.e. returns only predictions, no gradient):
def black_box_model(img,device,model):
    t_img = torch.from_numpy(img).float().div(255).permute(2, 0, 1)
    t_img = t_img.unsqueeze(0).to(device)
    with torch.no_grad():
        return model(t_img).argmax()

def attack(model,
           s_models: Iterable[torch.nn.Module],
           attacks,
           data,
           labels,
           targeted: bool,
           device: Optional[torch.device]):
    """ Run a black-box attack on 'model', using surrogate models 's_models'
        and attackers 'attacks'

        For each surrogate model and attacker, we create an attack against the
        surrogate model, and use the resulting direction to create an attack
        against the defense ('model'). This is done with a binary search along
        this direction.

    Parameters
    ----------
    model : defense model under attack
    s_models : List of torch models
        The surrogate models. Each model should be a PyTorch nn.Module, that
        takes an input x (B x H x W x C) with pixels from [0, 1], and returns
        the pre-softmax activations (logits).
    attacks : List of attack functions
        List of attacks. Each attack should have a method as follows:
            attack(model, inputs, labels, targeted) -> adv_image
    data : tensor
        A tensor (B x H x W x C) ranging from [0, 1]
    label : tensor(B x 1)
        The true label (if targeted=True) or target label (if targeted=False)
    targeted : bool
        Wheter to run untargeted or a targeted attack
    device : torch.device
        Which device to use for the attacks

    Returns
    -------
    batch_size x np.ndarray:
        The best adversarial image found against 'model'. None if no
        adversarial is found.

    """
    print(targeted)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = len(labels)
    print(batch_size)
    data = data.to(device)
    labels = labels.to(device)

    adversarial = None

    adv_batch = []
    #generate adv_batch
    for s_m in s_models:

        for attack in attacks:
            adv_img = attack.attack(s_m, data, labels, targeted)
            adv_batch.append(adv_img)
    # bound search and binary search via image on adv_batch
    adv_result = []
    for i in range(batch_size):
        image = data[i].permute(1, 2, 0).cpu().numpy() * 255
        label = labels[i]
        best_norm = np.linalg.norm(np.maximum(255 - image, image))


        for mm in range(len(adv_batch)):

            adv_img = adv_batch[mm]
            delta = (adv_img[i]-data[i]).permute(1, 2, 0).cpu().numpy() * 255
            delta = np.round(delta)
            norm = np.linalg.norm(delta)

            if norm > 0:
                # Run bound search
                lower, upper, found = bound_search(model, device, image,
                                                   label, delta,
                                                   targeted=targeted)
                if found:
                    norm = np.linalg.norm(upper)
                    if norm < best_norm:
                        adversarial = upper + image
                        best_norm = norm

                # Run binary search
                upper_, found_ = binary_search(model, device, image, label,
                                               lower, upper, steps=10,
                                               targeted=targeted)

                if found_:
                    norm = np.linalg.norm(upper_)
                    if norm < best_norm:
                        adversarial = upper_ + image
                        best_norm = norm
        adv_result.append(adversarial)
    #print(np.array(adv_result[20]))
    return adv_result



def bound_search(model, device, image, label, delta, alpha=1, iters=20, targeted=False):
    """ Coarse search for the decision boundary in direction delta """
    def out_of_region(delta):
        # Returns whether or not image+delta is outside the desired region
        # (e.g. inside the class boundary for untargeted, outside the target
        # class for targeted)
        pre_label = black_box_model(image + delta, device, model)
        if targeted:
            return pre_label != label
        else:
            return pre_label == label

    if out_of_region(delta):
        # increase the noise
        lower = delta
        upper = np.clip(image + np.round(delta * (1 + alpha)), 0, 255) - image

        for _ in range(iters):
            if out_of_region(upper):
                lower = upper
                adv = image + np.round(upper * (1 + alpha))
                upper = np.clip(adv, 0, 255) - image
            else:
                return lower, upper, True
    else:
        # inside the region of interest. Decrease the noise
        upper = delta
        lower = np.clip(image + np.round(delta / (1 + alpha)), 0, 255) - image

        for _ in range(iters):
            if not out_of_region(lower):
                upper = lower
                adv = image + np.round(lower / (1 + alpha))
                lower = np.clip(adv, 0, 255) - image
            else:
                return lower, upper, True

    return np.zeros_like(delta), np.round(delta / delta.max() * 255), False


def binary_search(model, device, image, label, lower, upper, steps=20, targeted=False):
    """ Binary search for the decision boundary in direction delta """
    def out_of_region(delta):
        # returns whether or not image+delta is outside the desired region
        # (e.g. inside the class boundary for untargeted, outside the target
        # class for targeted)
        pre_label = black_box_model(image + delta, device, model)
        if targeted:
            return pre_label != label
        else:
            return pre_label == label

    found = False
    for _ in range(steps):
        middle = np.round((lower + upper) / 2)
        middle = np.clip(image + middle, 0, 255) - image
        if out_of_region(middle):
            lower = middle
        else:
            upper = middle
            found = True

    return upper, found
