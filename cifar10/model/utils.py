"""This module implements functions used by DiStyleGAN."""
import json
from pathlib import Path

import torch
import torchvision.transforms as T
from torch import nn


def save_images(images: torch.Tensor, path: str, epoch: int) -> None:
    """Save a single or a batch of images.

    This method takes as input a tensor containing the images to
    be saved, and saves them in the following directory:
                `path`/images/epoch-`epoch`/

    Args:
        - images (Tensor) : tensor containing the images to be saved. The 
                           tensor must have the following format:
                           (number_of_images x C x H x W)
        - path (str) : directory's path to save the images
        - epoch (int) : the current epoch, which is used for naming purposes
    """
    loc = Path(
        path,
        f'images/epoch-{epoch}'
    )
    loc.mkdir(parents=True, exist_ok=True)

    transform = T.ToPILImage()

    for i, img in enumerate(images):
        img = img.cpu()
        img_PIL = transform(img)

        filepath = loc / Path(f'image-{i}.png')
        img_PIL.save(filepath)


def save_checkpoints(
        netG: nn.Module, netD: nn.Module,
        optimizerG: torch.optim.Adam, optimizerD: torch.optim.Adam,
        epoch: int, path: str,
        logG: dict, logD: dict) -> None:
    """Save the Generator's and Discriminator's model states.

    This method saves the Generator and Discriminator states, along with the
    states of their corresponding optimizers and a log of their respective
    losses at that `epoach`, to the following files:

        - generator.pt
        - discriminator.pt
        - optimizerG.pt
        - optimizerD.pt
        - log.json

    inside the `path`/checkpoints/epoch-`epoch`/ directory.

    Args:
        - netG (nn.Module) : the Generator to be saved
        - netsD (nn.Module) : the Discriminator to be saved
        - epoch (int) : the current epoch, which is used for
                        naming purposes
        - path (str) : the directory's path to save the checkpoints
        - logG (dict) : dictionary of the Generator's losses
        - logD (dict) : dictionary of the Discriminator's losses
    """
    loc = Path(
        path,
        f'checkpoints/epoch-{epoch}'
    )
    loc.mkdir(parents=True, exist_ok=True)

    torch.save(
        netG.state_dict(),
        Path(loc, f'generator.pt')
    )

    torch.save(
        netD.state_dict(),
        Path(loc, f'discriminator.pt')
    )

    torch.save(
        optimizerG.state_dict(),
        Path(loc, f'optimizerG.pt')
    )

    torch.save(
        optimizerD.state_dict(),
        Path(loc, f'optimizerD.pt')
    )

    log = {
        "epoch": epoch,
        "lossG": logG,
        "lossD": logD
    }

    with open(Path(loc, "log.json"), "w") as f:
        json.dump(log, f)


def decay_lr(
        optimizer: torch.optim.Adam, max_epoch: int,
        initial_decay: int, initial_lr: float) -> None:
    """Gradually decay the `optimizer`'s learning rate.

    Args:
        - optimizer (Adam) : the optimizer whose learning rate is going to be
                             decayed
        - max_epoch (int) : the total epochs specified for the training
        - initial_decay (int) : the batch iteration at which the decay starts
        - initial_lr (float) : the initial learning rate of the optimizer
    """
    coeff = -initial_lr / (max_epoch - initial_decay)
    for pg in optimizer.param_groups:
        pg['lr'] += coeff