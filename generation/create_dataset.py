"""This module creates a dataset of synthetic images generated by StyleGAN2
for the CIFAR-10 dataset, to be used for knowledge distillation.

In particular it uses the official PyTorch implementation of the StyleGAN2-ADA
model by NVIDIA Research Projects on Github, along with the provided weights
for the pre-trained model on CIFAR-10, for conditional image generation.
(https://github.com/NVlabs/stylegan2-ada-pytorch).
"""
import argparse
import functools
import pickle
import random
from pathlib import Path

import PIL.Image
import torch
from tqdm import tqdm

# For reproducibility of the dataset creation
random.seed(42)


def create_dataset(args: argparse.Namespace) -> None:
    """ Create a dataset of CIFAR-10 synthetic images, generated by StyleGAN2.

    This function creates a dataset of `args.nsamples` synthetic samples for
    each of the 10 classes of the CIFAR-10 dataset, generated using a
    pre-trained StyleGAN2 model for conditional image generation on the same
    dataset. The random noise vectors used as input for the generation of the
    synthetic images are also saved, to be used for knowledge distillation.

    Args:
        - args.path (str) : path to save the generated images to
        - args.checkpoint (str) : path to StyleGAN2\'s checkpoint for CIFAR-10
        - args.nsamples (int) : number of samples to generator per class
        - args.batch_size (int) : number of samples per batch for the generator
        - args.device (str) : device to use for the image generation
    """
    # Set path for checkpoints
    root = Path(args.path)
    root.mkdir(parents=True, exist_ok=True)

    # Set device
    if args.device == None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Load StyleGAN2's pre-trained model for CIFAR-10
    with open(args.checkpoint, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)

    if device == 'cpu':
        G.forward = functools.partial(G.forward, force_fp32=True)

    # Indices for classes in CIFAR-10
    class_idx = range(0, 10)

    # Number of batches per class
    last_batch_size = args.nsamples % args.batch_size
    batches = args.nsamples // args.batch_size
    if last_batch_size != 0:
        batches += 1

    # Progress bar
    with tqdm(total=10 * batches) as pbar:
        for idx in class_idx:
            # Make dir for the class
            path = root / Path(f'class_{idx}')
            path.mkdir(parents=True, exist_ok=True)

            # Set labels
            label = torch.zeros([args.batch_size, G.c_dim], device=device)
            label[:, idx] = 1

            for batch in range(batches):
                z = torch.rand((args.batch_size, G.z_dim)).to(device)

                with torch.no_grad():
                    images = G(z, label, truncation_psi=1, noise_mode="const")

                images = (images.permute(0, 2, 3, 1)
                          * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                # For the last batch
                if batch == batches-1:
                    images = images[:last_batch_size]

                for i, img in enumerate(images):
                    # Save synthetic images
                    PIL.Image.fromarray(img.cpu().numpy(), 'RGB').save(
                        path / Path(f'image_{batch * args.batch_size + i}.png')
                    )
                    # Save noise vector
                    torch.save(
                        z[i],
                        path / Path(f'noise_{batch * args.batch_size + i}.pt'))
                pbar.update(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', required=True, type=str, dest='path',
                        help='Path to save the generated images to.')

    parser.add_argument('-c', '--checkpoint', required=True,
                        type=str, dest='checkpoint',
                        help='Path to StyleGAN2\'s checkpoint for CIFAR-10.')

    parser.add_argument('-n', '--nsamples', default=3000,
                        type=int, dest='nsamples',
                        help='Number of samples per class.')

    parser.add_argument(
        '-b', '--batch_size', default=32, type=int, dest='batch_size',
        help='Number of samples per minibatch.')

    parser.add_argument(
        '-d', '--device', default=None, choices=["cpu", "cuda"],
        type=str, dest='device', help='Device to use for the image generation')

    args = parser.parse_args()

    create_dataset(args)
