# CS570-Final-Project --Multi-Labeling ImageNet

## Problem Statement
ImageNet, as the most well-known image classification benchmark, suffers from the absence of multilabel support. Some existing works have tried to address this problem using machine annotators, heavily relying on external datasets.

## Motivation
Our goal is to relabel ImageNet to enable multi-label support, utilizing only the existing data within the ImageNet dataset.

## Proposed Methodology
Our method for multi-labeling ImageNet consists of three steps: **Cropping, Confidence, and Label**.

![alt text](https://github.com/mirfana23/CS570-Final-Project/blob/main/methodology.png?raw=true)


### Cropping
In this stage, we use Random Crop and Region Proposal Network to extract different regions from the images.

### Confidence
We estimate the confidence of each cropped region belonging to different classes. We use multiple forward approaches including MC dropout and MC perturbation.

### Label
In this stage, we have the confidence score for each cropped region, and we aim to pseudo label them. We use two approaches: Threshold-based classification and Gaussian Mixture Model.

After a pass of this relabelling process, we re-train the model with the multiple labels and evaluate it on the validation set. If the performance increases, we re-label the training set. The process is iterative and stops once the validation results start to converge.

## Experiment Settings
- **Backbone**: ResNet-50
- **Original Dataset**: Imagenet-1K
- **Test Dataset**: Imagenet-1K validation set (single-label), Imagenet2012_multilabel (multi-label), Imagenet_Real (multi-label)
- **Metric**: Single-label accuracy (Top-1 / Top-5 accuracy)
