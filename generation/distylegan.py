"""This module implements the DiStyleGAN model, which constitutes a
distillation attempt for the official PyTorch implementation of the
StyleGAN2-ADA model by NVIDIA Research Projects on Github
(https://github.com/NVlabs/stylegan2-ada-pytorch), for the task of
conditional image generation on CIFAR-10.
"""
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model.loss import DLoss, GLoss
from model.network import Discriminator, Generator
from model.utils import *


class DiStyleGAN(object):
    """DiStyleGAN class for the conditional image generation.

    Args:
        - c_dim (int, optional) : condition dimension (Default: 10)
        - z_dim (int, optional) : noise dimension (Default: 512)
        - ngf (int, optional) : number of generator filters in the first
                                convolutional layer (Default: 256)
        - ndf (int, optional) : number of discriminator filters in the first
                                convolutional layer (Default: 128)
        - lambda_pixel (float, optional) : weight for the pixel loss of the
                                           Generator (Default: 0.2)
        - lambda_ganG (float, optional) : weight for the adversarial
                                          distillation loss of the Generator
                                          (Default: 0.01)
        - lambda_ganD (float, optional) : weight for the adversarial GAN loss
                                          of the Discriminator (Default: 0.2)
        - project_dim (int, optional) : dimension to project the input
                                        condition (Default: 128)
        - nc (int, optional): number of channels for the images (Default: 3)
        - transform (callable, optional) : optional transform to be applied
                                           on a sample image
                                           (Default: None)
        - num_test (int, optional): number of generated images for evaluation
                                    (Default: 30)
        - device (str, optional): device to use for training ('cpu' or 'cuda')
                                  (Default: If there is a CUDA device
                                  available, it will be used for training)
    """

    def __init__(
        self,
        c_dim: int = 10,
        z_dim: int = 512,
        ngf: int = 256,
        ndf: int = 128,
        lambda_pixel: float = 0.2,
        lambda_ganG: float = 1e-2,
        lambda_ganD: float = 0.2,
        project_dim: int = 128,
        nc: int = 3,
        transform: Callable = None,
        num_test: int = 30,
        device: str = None,
    ) -> None:
        """ Initialize the DiStyleGAN model.

        Args:
        - c_dim (int, optional) : condition dimension (Default: 10)
        - z_dim (int, optional) : noise dimension (Default: 512)
        - ngf (int, optional) : number of generator filters in the first
                                convolutional layer (Default: 256)
        - ndf (int, optional) : number of discriminator filters in the first
                                convolutional layer (Default: 128)
        - lambda_pixel (float, optional) : weight for the pixel loss of the
                                           Generator (Default: 0.2)
        - lambda_ganG (float, optional) : weight for the adversarial
                                          distillation loss of the Generator
                                          (Default: 0.01)
        - lambda_ganD (float, optional) : weight for the adversarial GAN loss
                                          of the Discriminator (Default: 0.2)
        - project_dim (int, optional) : dimension to project the input
                                        condition (Default: 128)
        - nc (int, optional): number of channels for the images (Default: 3)
        - transform (callable, optional) : optional transform to be applied
                                           on a sample image (Default: None)
        - num_test (int, optional): number of generated images for evaluation
                                    (Default: 30)
        - device (str, optional): device to use for training ('cpu' or 'cuda')
                                  (Default: If there is a CUDA device
                                  available, it will be used for training)
        """
        self.num_test = num_test

        if not device:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        if transform is None:
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.transform = transform

        # Create configuration dictionary
        self.config = {
            "c_dim": c_dim,
            "z_dim": z_dim,
            "ngf": ngf,
            "ndf": ndf,
            "lambda_pixel": lambda_pixel,
            "lambda_ganG": lambda_ganG,
            "lambda_ganD": lambda_ganD,
            "project_dim": project_dim,
            "nc": nc
        }

    def __init_weights(self, m) -> None:
        """Initialize the weights.

        This method is applied to each layer of the Generator's and
        Discriminator's layers to initiliaze their weights and biases.
        """
        classname = m.__class__.__name__
        if isinstance(m, nn.Conv2d) or classname.find('Conv') != -1:
            nn.init.orthogonal_(m.weight.data, 1.0)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            nn.init.orthogonal_(m.weight.data, 1.0)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def load_networks(
            self, checkpoint: str = None) -> "Tuple[Generator, Discriminator]":
        """Load the generator and discriminator networks.

        Args:
            - checkpoint (str, optional) : path to checkpoint's files
                                           (Default: None)
        """
        netG = Generator(
            self.config["ngf"], self.config["z_dim"], self.config["c_dim"],
            self.config["project_dim"], self.config["nc"])

        netD = Discriminator(
            self.config["ndf"],
            self.config["c_dim"],
            self.config["project_dim"],
            self.config["nc"])

        if checkpoint:
            try:
                gen_path = Path(checkpoint, 'generator.pt')
                netG.load_state_dict(torch.load(
                    gen_path, map_location=self.device))

                dis_path = Path(checkpoint, 'discriminator.pt')
                netD.load_state_dict(torch.load(
                    dis_path, map_location=self.device))

            except FileNotFoundError:
                print("[ERROR] Wrong checkpoint path or files don\'t exist.")
                exit(1)
        else:
            netG.apply(self.__init_weights)
            netD.apply(self.__init_weights)

        netG = netG.to(self.device)
        netD = netD.to(self.device)

        return netG, netD

    def __set_test(self, checkpoint=None) -> None:
        """Initialize the test set for evaluation.

        This method generates random noise test vectors and corresponding
        random-class test labels. In order to evaluate the performance of the
        model, this test set must be fixed since the beginning of the training.
        In case the training is resumed, the aforementioned vectors are loaded
        from the given `checkpoint` path.

        Args:
            - checkpoint (str, optional) : path to checkpoint's files
                                           (Default: None)
        """
        if checkpoint:
            test_labels_path = Path(checkpoint, 'labels.pt')
            test_z_path = Path(checkpoint, 'noise.pt')
            try:
                self.test_labels = torch.load(
                    test_labels_path, map_location=self.device)
                self.test_z = torch.load(test_z_path, map_location=self.device)
            except FileNotFoundError:
                print("[ERROR] Wrong checkpoint path or files don\'t exist.")
                exit(1)
            return

        self.test_z = torch.randn(
            self.num_test, self.config["z_dim"]).to(
            self.device)
        self.test_labels = torch.zeros(self.num_test, 10).to(self.device)
        for i in range(self.num_test):
            self.test_labels[i][random.randint(0, 9)] = 1

        torch.save(self.test_z, Path(self.save, "noise.pt"))
        torch.save(self.test_labels, Path(self.save, "labels.pt"))

    def __define_optimizers(
        self, G_lr: float, D_lr: float,
        adam_momentum: float = 0.5,
        checkpoint=None
    ) -> "Tuple[torch.optim.Adam, torch.optim.Adam]":
        """Define the optimizers.

        This method initializes the Adam optimizers for both the Generator
        and the Discriminator.

        Args:
            - G_lr (float) : learning rate for the Generator
            - D_lr (float) : learning rate for the Discriminator
            - adam_momentum (float, optional) : Adam momentum (Defualt: 0.5)
            - checkpoint (str, optional) : path to checkpoint's files
                                           (Default: None)
        """
        optimizerD = torch.optim.Adam(self.netD.parameters(),
                                      lr=D_lr,
                                      betas=(adam_momentum, 0.999))

        optimizerG = torch.optim.Adam(self.netG.parameters(),
                                      lr=G_lr,
                                      betas=(adam_momentum, 0.999))

        if checkpoint:
            optG_path = Path(checkpoint, 'optimizerG.pt')
            optD_path = Path(checkpoint, 'optimizerD.pt')
            try:
                optimizerG.load_state_dict(torch.load(
                    optG_path, map_location=self.device))
                optimizerD.load_state_dict(torch.load(
                    optD_path, map_location=self.device))
            except FileNotFoundError:
                print("[ERROR] Wrong checkpoint path or files don\'t exist.")
                exit(1)

        return optimizerG, optimizerD

    def __evaluate(self, path: str, epoch: int) -> None:
        """Generate images for the `num_test` test samples selected.

        This method is called at the end of each training epoch in order to
        evaluate the performance of the model during training, by generating
        and saving images based on the test set's noise and labels.

        Args:
            - path (str) : path to save the images
            - epoch (int) : the current epoch of training
        """
        self.netG.eval()
        with torch.no_grad():
            images = self.netG(self.test_z, self.test_labels)

        dirname = f"epoch-{epoch}"
        save_images(images, path, dirname)

    def update_D(
            self, z: torch.Tensor, teacher_image: torch.Tensor,
            label: torch.Tensor, real_image: torch.Tensor,
            real_label: torch.Tensor) -> "tuple[dict, list]":
        """Update the Discriminator network's parameters.

        This method is called at each training iteration to update the
        Discriminator network's parameters. It returns a log of the losses, 
        and a list of features extracted by the discriminator for the 
        `teacher_image`.

        Args:
            - z (torch.Tensor) : noise tensor from the FakeCIFAR10 dataset
            - teacher_image (torch.Tensor) : the fake image generated by the
                                             teacher network that corresponds
                                             to the input noise tensor `z`
            - label (torch.Tensor) : the label that corresponds to the input
                                     image `teacher_image`
            - real_image (torch.Tensor) : a sample image from the official
                                          CIFAR10 dataset
            - real_label (torch.Tensor) : the label that corresponds to the
                                          input `real_image`

        """
        for param in self.netD.parameters():
            param.requires_grad = True

        # Knowledge Distillation - Teacher
        dis_teacher, features_teacher = self.netD(teacher_image, label)
        features_teacher = [h.detach() for h in features_teacher]

        # Knowledge Distillation - Student
        student_image = self.netG(z, label).detach()
        dis_student, _ = self.netD(student_image, label)

        # Adversarial Loss - Real
        dis_real, _ = self.netD(real_image, real_label)

        # Adversarial Loss - Random
        noise = torch.randn(
            self.batch_size, self.config["z_dim"]).to(
            self.device)
        random_image = self.netG(noise, real_label).detach()
        dis_random, _ = self.netD(random_image, real_label)

        lossD, logD = self.criterionD(dis_student, dis_teacher,
                                      dis_random, dis_real)

        self.optimizerD.zero_grad()
        lossD.backward()
        self.optimizerD.step()

        return logD, features_teacher

    def update_G(self, z: torch.Tensor, teacher_image: torch.Tensor,
                 features_teacher: "list[torch.Tensor]", label: torch.Tensor,
                 real_label: torch.Tensor, adversarial: bool) -> dict:
        """Update the Generator network's parameters.

        This method is called at each training iteration to update the
        Generator network's parameters. It returns a log of the losses.

        Args:
            - z (torch.Tensor) : noise tensor from the FakeCIFAR10 dataset
            - teacher_image (torch.Tensor) : the fake image generated by the
                                             teacher network that corresponds
                                             to the input noise tensor `z`
            - features_teacher (list[torch.Tensor]): a list of features 
                extracted by the discriminator for the `teacher_image`
            - label (torch.Tensor) : the label that corresponds to the input
                                     image `teacher_image`
            - real_label (torch.Tensor) : the labels used in the current
                                          batch of real images
            - adversarial (bool) : defines whether or not to use the 
                                   adversarial losses in the current update

        """
        self.netG.train()

        for param in self.netD.parameters():
            param.requires_grad = False

        student_image = self.netG(z, label)
        dis_student, features_student = self.netD(student_image, label)

        if adversarial:
            noise = torch.randn(
                self.batch_size, self.config["z_dim"]).to(
                self.device)
            random_image = self.netG(noise, real_label)
            dis_random, _ = self.netD(random_image, real_label)
            lossG, logG = self.criterionG(
                student_image,
                teacher_image,
                features_student,
                features_teacher,
                dis_student,
                dis_random
            )
        else:
            lossG, logG = self.criterionG(
                student_image,
                teacher_image,
                features_student,
                features_teacher
            )

        self.optimizerG.zero_grad()
        lossG.backward()
        self.optimizerG.step()

        return logG

    def train(
        self,
        dataset: str,
        save: str,
        real_dataset: str = None,
        epochs: int = 150,
        batch_size: int = 128,
        gstep: int = 10,
        lr_G: float = 0.0002,
        lr_D: float = 0.0002,
        adam_momentum: float = 0.5,
        lr_decay: int = 350000,
        checkpoint_interval: int = 20,
        checkpoint_path: str = None,
        num_workers: int = 0
    ):
        """Train DiStyleGAN.

        Args:
            - dataset (str) : path to the dataset directory of the fake CIFAR10 
                              data generated by the teacher network
            - save (str) : path to save checkpoints and results
            - real_dataset (str, optional) : path to the dataset directory of 
                                             the real CIFAR10 data.
                                             (Default: None, it will be
                                             downloaded and saved in the parent
                                             directory of input `dataset` path)
            - epochs (int, optional) : number of training epochs
                                       (Default: 150)
            - batch_size (int, optional) : number of samples per batch
                                          (Default: 128)
            - gstep (int, optional) : the number of discriminator updates
                                      after which the generator is updated
                                      using the full loss (Default: 10)
            - lr_G (float, optional) : learning rate for the generator's
                                       Adam optimizers (Default: 0.0002)
            - lr_D (float, optional) : learning rate for the discriminator's
                                       Adam optimizers (Default: 0.0002)
            - adam_momentum (float, optional) : momentum value for the
                                                Adam optimizers' betas 
                                                (Default: 0.5)
            - lr_decay (int, optional) : iteration to start decaying the 
                                         learning rates for the Generator and 
                                         the Discriminator (Default: 350000)
            - checkpoint_interval (int, optional) : checkpoints will be saved
                                                    every `checkpoint_interval` 
                                                    epochs (Default: 20)
            - checkpoint_path (str, optional) : path to previous checkpoint
            - num_workers (int, optional) : number of subprocesses to use
                                            for data loading (Default: 0, 
                                            whichs means that the data will be 
                                            loaded in the main process.)
        """
        date = datetime.now().strftime("%d-%b-%Y (%H.%M)")
        self.save = Path(save, date)
        self.save.mkdir(exist_ok=True, parents=True)

        self.batch_size = batch_size

        if checkpoint_path is not None:
            with open(Path(checkpoint_path, "config.json"), "r") as f:
                self.config = json.load(f)

        # Save network's configuration
        with open(Path(self.save, "config.json"), "w") as f:
            json.dump(self.config, f)

        # Tensorboard
        writer = SummaryWriter(Path(self.save, "tensorboard"))

        # Load the networks
        self.netG, self.netD = self.load_networks(checkpoint_path)

        # Get dataloaders
        fakeCIFAR_loader, cifar10_loader = get_dataloaders(
            dataset, self.transform, self.batch_size,
            real_dataset, num_workers)

        # Set test dataset
        self.__set_test(checkpoint_path)

        # Optimizers
        self.optimizerG, self.optimizerD = self.__define_optimizers(
            lr_G, lr_D, adam_momentum, checkpoint_path)

        # Define criteria
        self.criterionG = GLoss(
            self.config["lambda_pixel"],
            self.config["lambda_ganG"])
        self.criterionD = DLoss(self.config["lambda_ganD"])

        training_start = datetime.now()
        print(
            f"\n{training_start.strftime('%d %B [%H:%M:%S] ')}"
            "Starting training..."
        )

        if checkpoint_path:
            with open(Path(checkpoint_path, "log.json"), "r") as f:
                log = json.load(f)
            # Resume training from the previous epoch
            starting_epoch = log["epoch"] + 1
        else:
            starting_epoch = 1

        total_iterations = len(fakeCIFAR_loader) * epochs

        for epoch in range(starting_epoch, epochs+1):
            real_iter = iter(cifar10_loader)
            for i, (z, teacher_image, label) in enumerate(fakeCIFAR_loader):
                # Decay the learning rates
                if (epoch-1)*len(fakeCIFAR_loader) + i >= lr_decay:
                    decay_lr(self.optimizerG, total_iterations, lr_decay, lr_G)
                    decay_lr(self.optimizerD, total_iterations, lr_decay, lr_D)

                z = z.to(self.device)
                teacher_image = teacher_image.to(self.device)
                label = label.to(self.device)

                # Get real data
                try:
                    real_image, real_label = next(real_iter)
                except:
                    real_iter = iter(cifar10_loader)
                    real_image, real_label = next(real_iter)

                # Convert CIFAR-10 labels to one-hot encoding
                real_label = F.one_hot(
                    real_label, num_classes=10).float()

                real_image = real_image.to(self.device)
                real_label = real_label.to(self.device)

                # Update Discriminator
                logD, features_teacher = self.update_D(
                    z, teacher_image, label, real_image, real_label)

                # Update Generator
                adversarial = (i+1) % gstep == 0
                logG = self.update_G(
                    z, teacher_image, features_teacher, label, real_label,
                    adversarial)

                # Print training information
                print(
                    f"Epoch: [{epoch}/{epochs}] \t"
                    f"Batch [{i + 1}/{len(fakeCIFAR_loader)}] \t Generator "
                    f"Loss: {logG['G/Total Loss']:.5f} \t Discriminator "
                    f"Loss: {logD['D/Total Loss']:5f}", end="\r"
                )

            # Tensorboard logging
            for key, value in logG.items():
                writer.add_scalar(key, value, epoch)
            for key, value in logD.items():
                writer.add_scalar(key, value, epoch)

            self.__evaluate(self.save, epoch)
            if (epoch % checkpoint_interval == 0 or epoch == epochs):
                save_checkpoints(
                    self.netG, self.netD,
                    self.optimizerG, self.optimizerD,
                    epoch, self.save, logG, logD)

        training_end = datetime.now()
        print(
            f"\n{training_end.strftime('%d-%b [%H:%M:%S] ')}"
            "Finished training."
        )
        duration = (training_end - training_start)
        print(
            "Training duration: "
            f"{duration.days} days, {duration.seconds // 3600} hours"
            f" and {(duration.seconds // 60) % 60} minutes"
        )

    def generate(
        self,
        checkpoint_path: str,
        nsamples: int,
        label: "int | list[int]" = None,
        save: str = None,
        batch_size: int = 32
    ) -> torch.Tensor:
        """Generate images using a pre-trained model's checkpoint.

        Args:
            - checkpoint_path (str) : path to previous checkpoint (the
                directory must contain the generator.pt and config.json files)
            - nsamples (int) : number of samples to generate
            - label (int, list[int], optional) : class label for the samples
                                                 (Default: None, random labels)
            - save (str) : path to save the generated images (Default: None)
            - batch_size (int, optional) : number of samples per batch
                                          (Default: 32)
        """
        with open(Path(checkpoint_path, "config.json"), "r") as f:
            self.config = json.load(f)

        # Load the generator
        netG = Generator(
            self.config["ngf"], self.config["z_dim"], self.config["c_dim"],
            self.config["project_dim"], self.config["nc"])
        gen_path = Path(checkpoint_path, 'generator.pt')
        netG.load_state_dict(torch.load(
            gen_path, map_location=self.device))
        netG = netG.to(self.device)
        netG.eval()

        if isinstance(label, int):
            label = [label]
        elif label is None:
            label = ["random"]

        if nsamples < batch_size:
            batch_size = nsamples

        all_images = []
        for l in label:
            print(f"Generating images for label {l}...")
            images = torch.Tensor()
            labels = torch.zeros(batch_size, 10).to(self.device)
            if l == "random":
                for k in range(batch_size):
                    labels[k][random.randint(0, 9)] = 1
            else:
                labels[:, l] = 1
            for i in range(nsamples // batch_size + 1):
                # Last batch
                if i == (nsamples // batch_size):
                    last_size = nsamples % batch_size
                    labels = torch.zeros(last_size, 10).to(self.device)
                    if l == "random":
                        for k in range(last_size):
                            labels[k][random.randint(0, 9)] = 1
                    else:
                        labels[:, l] = 1

                    noise = torch.randn(last_size, self.config["z_dim"]).to(
                        self.device)
                else:
                    noise = torch.randn(batch_size, self.config["z_dim"]).to(
                        self.device)

                with torch.no_grad():
                    images = torch.cat(
                        [images, netG(noise, labels)],
                        dim=0)

            all_images.append(images)

            if save is not None:
                save_images(images, save, f"class-{l}")

        all_images = torch.stack(all_images)

        return all_images
