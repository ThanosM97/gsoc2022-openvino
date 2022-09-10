<h1 align="center"> Google Summer of Code 2022: Train a DL model for synthetic data generation for model optimization </h1>

This repository hosts the code for the GSoC22 project 
["Train a DL model for synthetic data generation for model optimization"](https://summerofcode.withgoogle.com/programs/2022/projects/bCmHAPIL), 
which is implemented under the auspices of Intel's [OpenVINO Toolkit](https://github.com/openvinotoolkit) organization.

## About the project
The project is divided into two parts. For the first part, the goal is to train a lightweight Deep Learning model to generate synthetic images. For the second part, the pre-trained model of the first part is used to generate a dataset of synthetic images for CIFAR-10. Subsequently, this dataset is used for model optimization with [OpenVINO's Post-training Optimization Tool](https://docs.openvino.ai/latest/pot_introduction.html). We evaluate the performance of the 8-bit post-training quantization method on a range of Computer Vision models.

### Model training (PART I)
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

The goal of this part of the project is to investigate whether or not it is possible to generate quality images, using a lightweight model, by leveraging the aforementioned distillation framework. For our experiments we use the CIFAR-10 [[8]](#8) dataset, which although has a small spatial size (32x32), it is still complex enough to require a large model to generate quality images. The selected teacher network for the distillation procedure is the StyleGAN2-ADA [[9]](#9) model, which has more than 20 million trainable parameters. Its official PyTorch implementation can be found [here](https://github.com/NVlabs/stylegan2-ada-pytorch).


### Post-training Model Optimization (PART II)
Post-training model optimization is the task of applying techniques, such as post-training 8-bit quantization, to improve the performance of a model, without the need for retraining or finetuning. Using OpenVINO toolkit, the optimization procedure does not require a training dataset, but rather a representative calibration dataset of a relatively small number of samples (e.g. 300 samples).

The goal of the second part of the project is to evaluate the performance of the 8-bit post-training quantization method, on a range of Computer Vision models. We compare the performance of the optimized models when a subset of the original CIFAR-10 dataset is used to calibrate the model, with the corresponding performance using a calibration dataset of synthetic generated images by the trained DL model of the first part of the project.

## Overview

* [Home](.)
* [Timeline](../../wiki/Timeline)
* PART I
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
        * [Training](../../wiki/Training)
        * [Image Generation](../../wiki/Image-Generation)
        * [Evaluation](../../wiki/Evaluation)
* PART II
    * [Model Optimization](../../wiki/Model-Optimization)
    * [OpenVINO Toolkit](../../wiki/OpenVINO)
    * [Experiments](../../wiki/Experiments)
        * [PyTorch Models](../../wiki/PyTorch-Models)
        * [Calibration Datasets](../../wiki/Calibration-Datasets)
        * [Results](../../wiki/Results)



## Project Deliverables

* A synthetic [dataset](https://drive.google.com/file/d/1G6aGsUe7PWlRI9iO78u6_dlpoLiAFGZ-/view?usp=sharing) of 50,000 samples generated using the StyleGAN2-ADA model
* A pre-trained conditional GAN, DiStyleGAN, to generate images from the CIFAR-10 distribution ( [weights](https://drive.google.com/file/d/1Bjr7sQhVQzkYaIOVpx4KaBP3OyRxT5h1/view?usp=sharing) | [information](../../wiki/Student-Network) )
* A [webapp](./webapp) for conditional image generation using the pre-trained model
* Three calibration [datasets](https://drive.google.com/file/d/1e38vn-_VHMcGDkEUvTOUaaX8HuIkDGf9/view?usp=sharing) for model optimization
* A range of computer vision [quantized models](https://drive.google.com/file/d/1qFV-eE1omAFtpL13z5sIsjQhwJ1wrHwG/view?usp=sharing) for image classification on CIFAR-10
* Two blogs describing the [development of a class-conditional GAN for synthetic image generation](https://medium.com/openvino-toolkit/train-a-dl-model-for-synthetic-data-generation-for-model-optimization-with-openvino-part-1-314252de7148) and the [quantization of deep learning models using OpenVINO Toolkit](https://medium.com/openvino-toolkit/train-a-dl-model-for-synthetic-data-generation-for-model-optimization-with-openvino-part-2-fe1e0fb68400)

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

