"""This module creates a GIF showcasing the progress of training by visualizing
the evolution of the test samples generated by the model for each epoch.
"""
import argparse
from pathlib import Path

import numpy as np
import PIL
import torchvision
from matplotlib import pyplot as plt
from PIL import ImageDraw, ImageFont, ImageOps
from torchvision.io import read_image
from torchvision.utils import make_grid
from tqdm import tqdm


def add_title(img: PIL.Image, title: str) -> PIL.Image:
    """Add the `title` to the `img` given as input.

    Args:
        - img (PIL.Image) : image to add title to
        - title (str) : the title to add to the image
    """
    img = ImageOps.expand(img, border=(5, 20, 5, 5), fill=(255, 255, 255))
    width, _ = img.size
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("FONTS/arial.ttf", 16)
    w = draw.textlength(title, font=font)
    draw.text(
        ((width - w) / 2, 0),
        title, fill=(0, 0, 0),
        font=font)

    return img


def create_gif(args: argparse.Namespace) -> None:
    """Create a GIF showcasing the progress of training.

    This module creates a GIF showcasing the progress of training by 
    visualizing the evolution of the test samples generated by the model for 
    each epoch.

    Args:
        - args.path (str) : path to the "images/" directory from training
        - args.save (str) : filename for the .gif file
        - args.duration (int) : GIF duration in seconds
    """
    directory = Path(args.path)

    grids = []
    epochs = list(directory.glob("*"))
    with tqdm(total=len(epochs)) as pbar:
        for i in range(1, len(epochs) + 1):
            images = []
            epoch = directory / Path(f"epoch-{i}")
            for img in epoch.glob("*"):
                images.append(read_image(str(img)))

            grid = make_grid(images, nrow=int(np.ceil(np.sqrt(len(images)))))
            grid_image = torchvision.transforms.ToPILImage()(grid)
            grid_image = add_title(grid_image, title=f"Epoch {i}")
            grids.append(grid_image)

            pbar.update(1)

    print("Saving the .gif file, this may take a while...")
    duration = int(args.duration * 1000 / len(grids))
    grids[0].save(
        f'{args.save}.gif', save_all=True, append_images=grids[1:],
        optimize=False, duration=duration, loop=1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', required=True, type=str, dest='path',
                        help='Path to the "images/" directory from training.')

    parser.add_argument('-s', '--save', required=True,
                        type=str, dest='save',
                        help='Filename for the .gif file.')

    parser.add_argument('-d', '--duration', default=15,
                        type=int, dest='duration',
                        help='GIF duration in seconds.')

    args = parser.parse_args()

    create_gif(args)
