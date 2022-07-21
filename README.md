<h1 align="center"> Google Summer of Code 2022: Development of a lightweight class-conditional GAN for image generation </h1>

This repository hosts the code for the GSoC22 project 
["Development of a lightweight class-conditional GAN for image generation"](https://summerofcode.withgoogle.com/programs/2022/projects/bCmHAPIL), 
which is implemented under the auspices of Intel's [OpenVINO Toolkit](https://github.com/openvinotoolkit) organization.

## About the project
Ιmage Generation is a task of Computer Vision that has been long researched in the literature. 
Studies leverage Generative Adversarial Networks (GANs) [[1]](#1), which when trained well, they can produce realistic synthetic
images, using only a random noise vector as input. Recent models [[2]](#2) are able to generate images that are indistinguishable 
from real ones, even in complex datasets such as ImageNet [[3]](#3). Although this is an interesting task, a more practical 
one, and also more complex, is the task of Conditional Image Generation. It refers to the task of Computer Vision, 
in which a generative model is used to synthesize realistic-looking images based on input conditions. The conditions 
could be attributes, text descriptions, or class labels, among others. Recent advances in this topic present models [[4]](#4)[[5]](#5) 
that are able to generate high-quality and high-fidelity images, but at the expense of millions of parameters that require 
substantial computational resources.

Acknowledging this shortcoming, ways to compress models' sizes have been researched. In Romero et al. (2014) [[6]](#6), the authors propose a Knowledge 
Distillation framework for network compression in which the knowledge of a pre-trained teacher network (big model) is imparted to a student 
network (small model). Furthermore, in Chang et al. (2020) [[7]](#7), the authors built on this technique and proposed a black-box knowledge distillation 
method designed for GANs. In this framework, the teacher network is used to generate a dataset of fake images from the distribution of its Generator,
that will subseqently be used to train the student network. This technique does not require any access to the internal states and features of the
teacher network, which also can be discarded upon the creation of the fake dataset. 

The goal of this project is to investigate whether or not it is possible to generate quality images, using a lightweight model, by levraging the aforementioned
distillation framework. For our experiments we used the CIFAR-10 [[8]](#8) dataset, which although has a small spatial size (32x32), it is still complex enough to require 
a large model to generate quality images. The selected teacher network for the distillation procedure is the StyleGAN2-ADA [[9]](#9) model, which has more than 20 million
trainable parameters. Its official PyTorch implementation can be found [here](https://github.com/NVlabs/stylegan2-ada-pytorch).


## Overview

* [Home](../../wiki/)
* Timeline
* Project
    * [Conditional Image Generation](../../wiki/Conditional-Image-Generation)
    * [Knowledge Distillation](../../wiki/Knowledge-Distillation-Framework)
    * [Problem Statement](../../wiki/Problem-Statement)
    * [Black-Box Distillation](../../wiki/Black-Box-Distillation)
    * [Objectives](../../wiki/Objectives)
    * CIFAR-10
        * [Dataset](../../wiki/CIFAR10)
        * [Teacher Network](../../wiki/Teacher-Network)
        * [Student Network](../../wiki/Student-Network)
* Code
    * Getting Started
    * Training
    * Evaluation



## Project Delivarables


## People

* GSoC22 Student: Athanasios Masouris ([ThanosM97](https://github.com/ThanosM97))
* Mentor: Mansi Sharma ([mansishr](https://github.com/mansishr))
* Mentor: Zhuo Wu ([zhuo-yoyowz](https://github.com/zhuo-yoyowz))



## References
<a id="1">[1]</a> Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems 27 (2014).

<a id="2">[2]</a> Sauer, Axel, Katja Schwarz, and Andreas Geiger. "Stylegan-xl: Scaling stylegan to large diverse datasets." arXiv preprint arXiv:2202.00273 1 (2022).

<a id="3">[3]</a> J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, ImageNet: A Large-Scale Hierarchical Image Database. IEEE Computer Vision and Pattern Recognition (CVPR), 2009.

<a id="4">[4]</a> Kang, Minguk, et al. "Rebooting acgan: Auxiliary classifier gans with stable training." Advances in Neural Information Processing Systems 34 (2021): 23505-23518.

<a id="5">[5]</a> Brock, Andrew, Jeff Donahue, and Karen Simonyan. "Large scale GAN training for high fidelity natural image synthesis." arXiv preprint arXiv:1809.11096 (2018).

<a id="6">[6]</a> Romero, Adriana, et al. "Fitnets: Hints for thin deep nets." arXiv preprint arXiv:1412.6550 (2014).

<a id="7">[7]</a> Chang, Ting-Yun, and Chi-Jen Lu. "Tinygan: Distilling biggan for conditional image generation." Proceedings of the Asian Conference on Computer Vision. 2020.

<a id="8">[8]</a> Krizhevsky, Alex, and Geoffrey Hinton. "Learning multiple layers of features from tiny images." (2009): 7.

<a id="9">[9]</a> Karras, Tero, et al. "Training generative adversarial networks with limited data." Advances in Neural Information Processing Systems 33 (2020): 12104-12114.

