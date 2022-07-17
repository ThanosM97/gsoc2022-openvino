"""
This module creates a t-SNE visualization of generated images 
to qualitatively examine the generator's distribution and evaluate the 
diversity in terms of the generated samples. In particular, it uses the
pre-trained VGG-19 model to extract features (4096d) of from the generated 
images, which are then compressed to 300d using the PCA algorithm.
Subsequently, the t-SNE algorithm is used to map those pca features to two
dimensions, which are then ploted in an image grid, with images placed in 
neighboring tiles based on their similarity distance.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import rasterfairy
import torch
from matplotlib.patches import Patch
from matplotlib.pyplot import imshow
from PIL import Image, ImageOps
from seaborn import color_palette
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vgg19
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm


def factors(num: int) -> "tuple[int,int]":
    """Return `num`'s two closer factors to its square root."""
    ny = int(np.ceil(np.sqrt(num)))
    nx = int(num/ny)
    while nx * ny != num:
        ny -= 1
        nx = int(num/ny)
    return nx, ny


def get_colors(nclasses: int) -> "tuple[color_palette, list]":
    """Return an sns color_palette and the corresponding RGB colors in [0,255].

    Args:
        - nclasses (int) : number of classes in the samples
    """
    palette = color_palette(n_colors=nclasses)
    colors = []
    for color in palette:
        rgb_color = [int(val*255) for val in color]
        colors.append(tuple(rgb_color))

    return palette, colors


def get_loader(path: str, batch_size: int) -> "tuple[DataLoader, int]":
    """Return a dataloader and the corresponding number of classes found.

    This function uses PyTorch's ImageFolder to create a dataset. It returns
    the dataloader and the number of classes found in `path`.

    Args:
        - path (str) : path to generated samples 
                       (format: dir/{class-0, class-1, ...}/image_X.png)
        - batch_size (int) : number of samples per batch
    """
    data = ImageFolder(path, transforms.ToTensor())
    nclasses = len(data.classes)
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    return dataloader, nclasses


def pca(features: torch.Tensor, components: int = 300) -> np.array:
    """Calculate and return PCA features for `components` dimensions.

    Args:
        - features (Tensor) : extracted features from VGG-19
        - components (int, optional) : number of components to use in PCA
                                       (Default: 300)
    """
    features = np.array(features)
    pca_model = PCA(n_components=components)
    pca_model.fit(features)
    pca_features = pca_model.transform(features)

    return np.array(pca_features)


def visualize(args: argparse.Namespace):
    """Create t-SNE visualization of the dataset in `path` directory.

    This function creates a t-SNE visualization of the dataset in `path`
    directory and saves it as `filename`.png .

    Args:
        - args.path (str) : path to the directory of the generated images. 
                            The directory should have the following format:
                                dir/{class-0, class-1, ...}/image_X.png)
        - args.filename (str) : filename for the .png file
        - args.nsamples (int) : number of samples to use from each class
        - args.title (str, optional) : title for the image 
                                       (Default: t-SNE visualization)
        - args.batch_size (int, optional) : number of samples per batch
                                            (Default: 32)
    """
    # Load pretrained vgg19 model
    model = vgg19(weights="VGG19_Weights.DEFAULT").eval()

    # Define feature extractor
    feature_extractor = create_feature_extractor(
        model, return_nodes=['classifier.3'])

    # Get dataloader
    dataloader, nclasses = get_loader(args.path, args.batch_size)

    total_samples = nclasses * args.nsamples

    if total_samples > len(dataloader.dataset):
        print((f"[ERROR] Not enough samples generated. (total_samples_found" +
               f"={len(dataloader.dataset)})"))
        exit(1)

    # Extract features using VGG19
    images = torch.Tensor()
    features = torch.Tensor()
    labels = torch.Tensor()

    batches = total_samples//args.batch_size
    with tqdm(total=batches, desc="Extracting features") as pbar:
        for i, (img, label) in enumerate(dataloader):
            with torch.no_grad():
                features = torch.cat(
                    [features, feature_extractor(img)['classifier.3']],
                    dim=0)
                images = torch.cat(
                    [images, img],
                    dim=0)
                labels = torch.cat(
                    [labels, label],
                    dim=0)

            if (i+1)*args.batch_size > total_samples:
                features = features[:total_samples]
                images = images[:total_samples]
                labels = labels[:total_samples]
                break

            pbar.update(1)

    # Get PCA features
    pca_features = pca(features)

    # TSNE transformation
    tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2,
                init="random", verbose=1).fit_transform(pca_features)

    # Image configurations
    palette, colors = get_colors(nclasses)
    nx, ny = factors(total_samples)
    tile_width = images.shape[-1] + 6  # + 6 pixels for the border
    tile_height = images.shape[-2] + 6
    full_width = tile_width * nx
    full_height = tile_height * ny

    # Assign to grid
    grid = rasterfairy.transformPointCloud2D(tsne, target=(nx, ny))
    grid_image = Image.new('RGB', (full_width, full_height))

    # Create image
    for img, label, grid_pos in zip(images, labels, grid[0]):
        idx_x, idx_y = grid_pos
        x, y = tile_width * idx_x, tile_height * idx_y

        # Add color borders to images for class identification
        tile = transforms.ToPILImage()(img)
        tile = ImageOps.expand(tile, border=3, fill=colors[int(label.item())])

        grid_image.paste(tile, (int(x), int(y)))

    plt.figure(figsize=(16, 12))
    imshow(grid_image)

    legend_elements = [Patch(facecolor="w", edgecolor=palette[i],
                             label=f'Class {i}') for i in range(nclasses)]
    plt.legend(
        handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title(args.title)
    plt.savefig(args.filename+".png")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="t-SNE visualization of generated samples")

    parser.add_argument(
        '-p', '--path', required=True, type=str, dest='path',
        help=('Path to the directory of the generated images. ' +
              'The directory should have the following format: ' +
              'dir/{class-0, class-1, ...}/image_X.png)'))

    parser.add_argument('-f', '--filename', required=True,
                        type=str, dest='filename',
                        help='Filename for the .png file.')

    parser.add_argument('-n', '--nsamples', required=True,
                        type=int, dest='nsamples',
                        help='Number of samples to use from each class.')

    parser.add_argument('-t', '--title', default='t-SNE visualization',
                        type=str, dest='title',
                        help='Title for the image.')

    parser.add_argument('-b', '--batch_size', default=32,
                        type=int, dest='batch_size',
                        help='Number of samples per batch.')

    args = parser.parse_args()

    visualize(args)
