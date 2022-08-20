import base64
import io
import sys

import numpy as np
import torch
import torchvision.transforms as T
from flask import Flask, render_template, request
from torchvision.utils import make_grid

sys.path.insert(0, '../generation')
from distylegan import DiStyleGAN

app = Flask(__name__)


def inverse_normalization(image: torch.Tensor) -> torch.Tensor:
    """Inverse normalization from [-1,1] to [0,1]."""
    # [-1,1] to [0,2]
    image = image + 1

    # [0,2] to [0,1]
    image = image - image.min()
    image_0_1 = image / (image.max() - image.min())

    return image_0_1


@app.route('/', methods=['GET', 'POST'])
def home():
    # CIFAR-10 classes
    classes = {
        0: "Airplane",
        1: "Automobile",
        2: "Bird",
        3: "Cat",
        4: "Deer",
        5: "Dog",
        6: "Frog",
        7: "Horse",
        8: "Ship",
        9: "Truck"
    }

    checked = []
    label = None
    if request.method == "POST":
        form = request.form
        checked = form.keys()

        label = [int(key) for key in checked]
        nsamples = 20 if len(label) > 1 else 64

        if len(label) == 0:
            label = None  # Random images

    if label == None:
        nsamples = 64

    # Generate random images
    distylegan = DiStyleGAN()
    images = distylegan.generate(
        "../checkpoint", nsamples=nsamples, label=label).cpu()
    transform = T.ToPILImage()

    # Create a grid of images for each selected class
    grid_list = []
    nrow = int(np.ceil(np.sqrt(len(images[0])))) if label is None or len(
        label) == 1 else 10
    for class_images in images:
        grid = make_grid(class_images, nrow=nrow)
        grid = inverse_normalization(grid)
        img_PIL = transform(grid)
        data = io.BytesIO()
        img_PIL.save(data, "JPEG")
        grid_list.append(base64.b64encode(data.getvalue()).decode('utf-8'))

    return render_template(
        "index.html", classes=classes, checked=checked,
        img_data=grid_list)
