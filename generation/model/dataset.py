"""This module implements a custom Dataset for synthetic images generated
for the CIFAR-10 dataset and their labels, along with the input noise vectors 
used by the corresponding generator.

We assume the following structure for the dataset directory:
- class_0
    - image_0.png
    - noise_0.pt\n
    ...
- class_1
    - image_0.png
    - noise_0.pt\n
    ... \n
... 

as created by the `../create_dataset.py` script.
"""
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FakeCIFAR10(Dataset):
    """Fake CIFAR-10 images dataset.

    This class implements a dataset for synthetic images generated for 
    the CIFAR-10 dataset and their labels, along with the input noise vectors 
    used by the corresponding generator.

    Args:
        - dataset (str) : path to the dataset directory
        - transform (callable, optional) : torchvision transforms for the 
                                           images
    """

    def __init__(
            self, dataset: str, transform: transforms.Compose = None) -> None:
        """Initialize a FakeCIFAR10 Dataset.

        Args:
            - dataset (str) : path to the dataset directory
            - transform (callable, optional) : torchvision transforms for the 
                                               images
        """
        super(FakeCIFAR10, self).__init__()

        self.dataset = Path(dataset)
        self.transform = transform

        classes = [path for path in self.dataset.glob("*") if path.is_dir()]
        self.images = [file for images in classes for file in images.glob(
            "*") if file.suffix in ['.png', '.jpg', '.jpeg']]

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.images)

    def __getitem__(
            self,
            idx: int) -> "Tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """Return a sample of the dataset.

        Each sample consists of a synthetic CIFAR-10 image and its label, along 
        with the random noise vector used as input in the corresponding 
        generator.
        """
        filepath = self.images[idx]

        image_path = str(filepath)

        _class = int(filepath.parent.name.split("_")[-1])
        label = torch.zeros([10])
        label[_class] = 1

        _id = filepath.stem.split("_")[1]
        noise_path = filepath.with_name(f"noise_{_id}.pt")

        image = Image.open(image_path).convert('RGB')
        noise = torch.load(noise_path)

        if self.transform:
            image = self.transform(image)

        return noise, image, label
