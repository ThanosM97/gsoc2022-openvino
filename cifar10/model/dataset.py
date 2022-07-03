"""This module implements a custom Dataset for synthetic images generated
for the CIFAR-10 dataset, along with the input noise vectors used by the
corresponding generator.

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
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image


class FakeCIFAR10(Dataset):
    """Fake CIFAR-10 images dataset.

    This class implements a dataset for synthetic images generated for 
    the CIFAR-10 dataset, along with the input noise vectors used by the
    corresponding generator.

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a sample of the dataset.

        Each sample consists of a synthetic CIFAR-10 image, along with the
        random noise vector used as input in the corresponding generator.
        """
        image_path = str(self.images[idx])

        _id = self.images[idx].stem.split("_")[1]
        noise_path = str(self.images[idx].with_stem(
            f"noise_{_id}").with_suffix(".pt"))

        image = read_image(image_path).float()
        noise = torch.load(noise_path)

        if self.transform:
            image = self.transform(image)

        return noise, image
