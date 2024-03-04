# VGG-Implementation

# Configuration

![VGG Configurations](./images/configurations.png)

"Very Deep Convolutional Networks for Large-Scale Image Recognition" by Karen Simonyan and Andrew Zisserman.

Paper: https://arxiv.org/pdf/1409.1556.pdf

# Architecture

![VGG Architecture](./images/architecture.png)

## Different day, same old story

GPU POOR !!!

Didn't train cause I don't have a powerful GPU. But the architecture is there for playing.

## Info

Run script below to checkout the model informations

```sh
python info.py
```

### VGG11 (A)

![vgg11](./images/vgg11.jpg)

### VGG11_LRN (A-LRN)

![vgg11_lrn](./images/vgg11_lrn.jpg)

### VGG13 (B)

![vgg13](./images/vgg13.jpg)

### VGG16_1 (C)
![vgg16_1](./images/vgg16_1.jpg)

### VGG16 (D)

![vgg16](./images/vgg16.jpg)

### VGG19 (D)
![vgg19](./images/vgg19.jpg)

## Usage

Before running the script, place your data directory location for both train and test data in `root_dir="{DIR}"` here at [dataloader.py](./dataloader/dataloader.py)

```sh
python train.py --epochs 74 --vgg vgg16
```
