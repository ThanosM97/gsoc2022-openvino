"""This module implements the losses for the Generator and Discriminator of
the proposed GAN for conditional image generation. 

Based on the work of Chang et al. (2020) (https://arxiv.org/abs/2009.13829),
the following losses are used.

The Generator loss consists of the following four terms:

    \mathcal{L}_S = \mathcal{L}_{KD\_feat} + \lambda_1 \mathcal{L}_{KD\_pix} + 
        \lambda_2\mathcal{L}_{KD\_S} + \lambda_3\mathcal{L}_{GAN\_S}

where,
    - Feature Loss: The feature-level distillation loss, which alleviates the
        blurriness of the generated images when only the pixel-level distance
        is used. A list of features for each of the images is extracted using
        the initial convolutional layers of the Discriminator. Then a weighted
        sum is calculated, giving emphasis on higher-level features.
    - Pixel Loss: The pixel-level distillation loss which forces the student
        network to mimic the functionality of the teacher network, by
        minimizing the pixel-level distance between the images generated by
        both networks.
    - Adversarial Distillation Loss: The adversarial distillation loss helps
        the Generator to produce images that are indinstinguishable from those
        of the teacher network.
    - Adversarial GAN Loss: The adversarial GAN loss helps the Generator to
        produce images that follow the distribution of the real data.

The Discriminator loss consists of the following two terms:

    \mathcal{L}_{D} = \mathcal{L}_{KD\_D} + \lambda_4 \mathcal{L}_{GAN\_D}

where,
    - Adversarial Distillation Loss: The adversarial distillation loss helps
        the Discriminator to differentiate between images from the distribution
        of the teacher network and the student network.
    - Adversarial GAN Loss: The adversarial GAN loss helps the Discriminator to
        differentiate between images from the distribution of student network
        and the distribution of the real data.

Both adversarial losses are calculated using the hinge version 
(https://arxiv.org/abs/1705.02894v2) instead of binary cross entropy.

More information about the two losses can be found in Chang et al. (2020).
"""
from typing import Tuple

import torch
from torch import nn


class GLoss(nn.Module):
    """
    The Generator loss consists of the following four terms:

    \mathcal{L}_S = \mathcal{L}_{KD\_feat} + \lambda_1 \mathcal{L}_{KD\_pix} + 
        \lambda_2\mathcal{L}_{KD\_S} + \lambda_3\mathcal{L}_{GAN\_S}

    where,
        - Feature Loss: The feature-level distillation loss, which alleviates 
            the blurriness of the generated images when only the pixel-level 
            distance is used. A list of features for each of the images is 
            extracted using the initial convolutional layers of the 
            Discriminator. Then a weighted sum is calculated, giving emphasis 
            on higher-level features.
        - Pixel Loss: The pixel-level distillation loss which forces the 
            student network to mimic the functionality of the teacher network, 
            by minimizing the pixel-level distance between the images generated
            by both networks.
        - Adversarial Distillation Loss: The adversarial distillation loss 
            helps the Generator to produce images that are indinstinguishable 
            from those of the teacher network.
        - Adversarial GAN Loss: The adversarial GAN loss helps the Generator to
            produce images that follow the distribution of the real data.

    Both adversarial losses are calculated using the hinge version 
    (https://arxiv.org/abs/1705.02894v2) instead of binary cross entropy.
"""

    def __init__(self, lambda_pixel: float, lambda_gan: float) -> None:
        super(GLoss, self).__init__()
        """ Initialize the Generator Loss.

        Args:
            - lambda_pixel (float) : weight for the pixel loss
            - lambda_gan (float) : weight for the adversarial distillation loss
        """
        self.lambda1 = lambda_pixel
        self.lambda2 = lambda_gan
        self.lambda3 = lambda_gan * 0.2

        self.criterion = nn.L1Loss()
        self.weights = [1.0/16, 1.0/8, 1.0/4, 1.0]

    def hinge(self, dloss: torch.Tensor) -> torch.Tensor:
        """ Generator's hinge version of the adversarial loss.

        More information in https://arxiv.org/abs/1705.02894v2
        """
        return -torch.mean(dloss)

    def forward(
            self, xs: torch.Tensor, xt: torch.Tensor,
            sfeatures: "list[torch.Tensor]", tfeatures: "list[torch.Tensor]",
            dis_student: torch.Tensor = None, dis_random: torch.Tensor = None
    ) -> "Tuple[torch.Tensor, dict]":
        """Calculate the total Generator loss.

        Args:
            xs (Tensor) : fake generated image by the generator of the student
                          network
            xt (Tensor) : fake generated image by the generator of the teacher
                          network
            sfeatures (list[Tensor]) : list of extracted features from the
                                       discriminator for the fake generated
                                       image by the student network
            tfeatures (list[Tensor]) : list of extracted features from the
                                       discriminator for the fake generated
                                       image by the teacher network
            dis_student (Tensor, optional) : the discriminator's output for the 
                                             fake generated image by the 
                                             student network from distillation
                                             (Default: None)
            dis_random (Tensor, optional) : the discriminator's output for the 
                                            fake generated image by the student 
                                            network from a random input noise 
                                            vector (Default: None)
        """
        # Initialize log for the losses
        log = {}

        # Pixel-level distillation loss
        pixel_loss = self.criterion(xs, xt)
        log["G/Pixel Loss"] = pixel_loss.item()

        # Feature-level distillation loss
        feature_loss = 0
        for i in range(len(sfeatures)):
            feature_loss += self.weights[i] * self.criterion(
                sfeatures[i], tfeatures[i])
        log["G/Feature Loss"] = feature_loss.item()

        if dis_student is not None and dis_random is not None:
            # Adversarial distillation loss
            dis_student = self.hinge(dis_student)
            log["G/Adversarial Distillation Loss"] = dis_student.item()

            # Adversarial GAN loss
            dis_random = self.hinge(dis_random)
            log["G/Adversarial GAN Loss"] = dis_random.item()

            total_loss = (
                feature_loss + self.lambda1 * pixel_loss +
                self.lambda2 * dis_student +
                self.lambda3 * dis_random)
            log["G/Total Loss"] = total_loss.item()
        else:
            total_loss = feature_loss + self.lambda1 * pixel_loss
            log["G/Total Loss"] = total_loss.item()

        # Gradually decay the pixel-level distillation loss (lambda1) to zero
        self.lambda1 = max(0.00, self.lambda1-0.01)

        return total_loss, log


class DLoss(nn.Module):
    """
    The Discriminator loss consists of the following two terms:

        \mathcal{L}_{D} = \mathcal{L}_{KD\_D} + \lambda_4 \mathcal{L}_{GAN\_D}

    where,
        - Adversarial Distillation Loss: The adversarial distillation loss 
            helps the Discriminator to differentiate between images from the 
            distribution of the teacher network and the student network.
        - Adversarial GAN Loss: The adversarial GAN loss helps the 
            Discriminator to differentiate between images from the distribution 
            of the student network and the distribution of the real data.

    Both adversarial losses are calculated using the hinge version 
    (https://arxiv.org/abs/1705.02894v2) instead of binary cross entropy.
    """

    def __init__(self, lambda_gan: float) -> None:
        """ Initialize the Discriminator Loss.

        Args:
            - lambda_gan (float) : weight for the adversarial GAN loss
        """
        super(DLoss, self).__init__()
        self.lambda4 = lambda_gan

    def hinge(self,
              dis_real: torch.Tensor,
              dis_fake: torch.Tensor) -> "Tuple[torch.Tensor, torch.Tensor]":
        """ Discriminator's hinge version of the adversarial loss.

        More information in https://arxiv.org/abs/1705.02894v2
        """
        dis_real = torch.mean(torch.relu(1. - dis_real))
        dis_fake = torch.mean(torch.relu(1. + dis_fake))
        return dis_real, dis_fake

    def forward(self,
                dis_student: torch.Tensor, dis_teacher: torch.Tensor,
                dis_random: torch.Tensor, dis_real: torch.Tensor
                ) -> "Tuple[torch.Tensor, dict]":
        """Calculate the total Discriminator loss.

        Args:
            dis_student (Tensor) : the discriminator's output for the fake 
                                   generated image by the student network 
                                   from distillation
            dis_teacher (Tensor) : the discriminator's output for the fake 
                                   generated image by the teacher network 
                                   from distillation
            dis_random (Tensor) : the discriminator's output for the fake 
                                  generated image by the student network 
                                  from a random input noise vector
            dis_real (Tensor) : the discriminator's output for the image
                                belonging in the distribution of the real 
                                data                                
        """
        # Initialize log for losses
        log = {}

        # Hinge loss for the distillation pairs
        dis_teacher, dis_student = self.hinge(dis_teacher, dis_student)
        log["D/Teacher Loss"] = dis_teacher.item()
        log["D/Student Loss"] = dis_student.item()

        # Hinge loss for pairs of random generated and a real images
        dis_real, dis_random = self.hinge(dis_real, dis_random)
        log["D/Real Loss"] = dis_real.item()
        log["D/Random Loss"] = dis_random.item()

        total_loss = dis_teacher + dis_student + self.lambda4 * (
            dis_real + dis_random)
        log["D/Total Loss"] = total_loss.item()

        return total_loss, log
