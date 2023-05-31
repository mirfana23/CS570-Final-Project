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

## How to Run the Code

### Structure of the Config File
The config file is a YAML file that contains all the parameters needed to run the code. The config files are located under the path: CS570-Final-Project/config. The current version of the config files contain the optimized hyperparameters for each setup. However, the followings should be set appropriately before running the code:

- **json_dir:**: Currently, it is set to `/root/CS570-Final-Project/datasets/imgnet1k_our/train0.json`, it stores the path to the json file that contains the original labels for the training set. The base path should be updated if the code is run in a different server.
- **num_workers:**: Currently, it is set to 0, it stores the number of workers for the dataloader. It should be updated based on the server's capacity.
- **relabeled_dataset_save_path:**: Currently, it is set to `/root/CS570-Final-Project/datasets/imgnet1k_our/train1_mc_dropout.json`, it stores the path to the json file that will be created after relabelling the training set. The base path should be updated if the code is run in a different server.
- **num_relabel:**: Currently, it is set to 100, it stores the number of images that will be relabelled in each iteration. It should be updated based on the time it will take to relabel the images.

### How to Relabel with Naive
From the main directory, run the following commands:
```
cd src/run
python relabeller.py --config ../../config/relabel_naive.yaml
```

### How to Relabel with MC Dropout
From the main directory, run the following commands:
```
cd src/run
python relabeller.py --config ../../config/relabel_mc_dropout.yaml
```

### How to Relabel with MC Perturbation
From the main directory, run the following commands:
```
cd src/run
python relabeller.py --config ../../config/relabel_mc_perturbation.yaml
```