# VGG-Implementation

"Very Deep Convolutional Networks for Large-Scale Image Recognition" by Karen Simonyan and Andrew Zisserman.

Paper: https://arxiv.org/pdf/1409.1556.pdf

## Configuration

<img src="./images/configurations.png" alt="VGG Configurations" style="width:100%;">

## Architecture

<img src="./images/architecture.png" alt="VGG Architecture" style="width:100%;">

## Different day, same old story

GPU POOR !!!

Didn't train cause I don't have a powerful GPU. But the architecture is there for playing.

## Info

Run script below to checkout the model informations

```sh
python info.py
```

### VGG11 (A)

<img src="./images/vgg11.jpg" alt="vgg11" style="width:100%;">

### VGG11_LRN (A-LRN)

<img src="./images/vgg11_lrn.jpg" alt="vgg11_lrn" style="width:100%;">

### VGG13 (B)

<img src="./images/vgg13.jpg" alt="vgg13" style="width:100%;">


### VGG16_1 (C)

<img src="./images/vgg16_1.jpg" alt="vgg16_1" style="width:100%;">

### VGG16 (D)

<img src="./images/vgg16.jpg" alt="vgg16" style="width:100%;">

### VGG19 (E)

<img src="./images/vgg19.jpg" alt="vgg19" style="width:100%;">

## Usage

Before running the script, place your data directory location for both train and test data in `root_dir="{DIR}"` here at [dataloader.py](./dataloader/dataloader.py)

```sh
python train.py --epochs 74 --vgg vgg16
```
