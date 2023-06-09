import os
import sys

# To handle relative importing
module_paths = [
    os.path.abspath(os.path.join('../..')),
]
for module_path in module_paths:
    if module_path not in sys.path:
        sys.path.append(module_path)

import argparse
import ast
import pickle
import time
import torch
import numpy as np
import copy
import yaml
from tqdm import tqdm
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from src.dataloaders.dataloader import ImgNetDataset
from torch.utils.data import DataLoader
from src.utils import Config, MetricAvg
from torchvision.models import resnet50, ResNet50_Weights

# Command line arguments, only need to provide config path
argparser = argparse.ArgumentParser()
argparser.add_argument('--config', type=str, default='/root/CS570-Final-Project/config/train_base.yaml')
args = argparser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = Config(**config)

# Initialize the model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Load the model from a checkpoint if specified
if config.load_from_checkpoint is not False:
    model.load_state_dict(torch.load(config.load_from_checkpoint))

# Move the model to the device
model = model.to(config.device)

# Initialize the data loaders
train_dataset = ImgNetDataset(json_path=config.train_dataset.json_dir, type='train')
train_loader = DataLoader(train_dataset, batch_size=config.train_dataset.batch_size, shuffle=True, num_workers=config.train_dataset.num_workers)

val_datasets = {
    val_dataset.name: ImgNetDataset(json_path=val_dataset.json_dir, type='val')
    for val_dataset in config.val_datasets
}

val_loaders = {
    val_dataset.name: DataLoader(val_datasets[val_dataset.name], batch_size=val_dataset.batch_size, shuffle=False, num_workers=val_dataset.num_workers)
    for val_dataset in config.val_datasets
}

# Initialize the optimizer
# NOTE: Subject to change, this is an example
optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)

# Initialize the loss function
# NOTE: Subject to change, this is an example
loss_fn = torch.nn.BCELoss()
sigmoid = torch.nn.Sigmoid()

# Initialize the scheduler
# NOTE: Subject to change, this is an example
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler.step_size, gamma=config.scheduler.gamma)

# TODO: Implement metric_avg for acc1 and acc5 too
# Initialize the training loop
for epoch in range(config.num_epochs):
    # # Training loop
    # # NOTE: Subject to change, this is an example

    metric_avg = MetricAvg(['avg_loss'])
    model.train()
    progress_bar = tqdm(train_loader)
    print(f'Training on {config.train_dataset.name}')
    for x, y in progress_bar:
        x = x.to(config.device)
        y = y.to(config.device)
        optimizer.zero_grad()
        y_hat = sigmoid(model(x))
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()

        metric_avg.update('avg_loss', loss.item(), 1) # NOTE: We send the batch-wise average loss here. If you wish to send individual losses, change this to loss.item() * batch_size, batch_size
        progress_bar.set_postfix(metric_avg.get_all())

    scheduler.step()
    print(f"Epoch {epoch}, metrics: {metric_avg.get_all()}")

    # Validation loop
    # NOTE: Subject to change, this is an example

    metric_avg = MetricAvg(['avg_loss', 'acc1', 'acc5'])
    model.eval()
    for val_dataset in config.val_datasets:
        metric_avg.reset_all()
        val_loader = val_loaders[val_dataset.name]
        print(f'Validating on {val_dataset.name}')
        progress_bar = tqdm(val_loader)
        with torch.no_grad():
            for x, y in progress_bar:
                x = x.to(config.device)
                y = y.to(config.device)

                # In ReaL, there are some images which does not have any labels. We remove those images from the dataset.
                y_sum = torch.sum(y, dim=1)
                x = x[y_sum > 0]
                y = y[y_sum > 0]

                y_hat = sigmoid(model(x))
                loss = loss_fn(y_hat, y)
                
                metric_avg.update('avg_loss', loss.item(), 1) # NOTE: We send the batch-wise average loss here. If you wish to send individual losses, change this to loss.item() * batch_size, batch_size
                
                # acc1
                max_idx = torch.argmax(y_hat, dim=1)
                b_idx = torch.arange(y.shape[0])
                acc1_sum = torch.sum(y[b_idx, max_idx] == 1).item()

                metric_avg.update('acc1', acc1_sum, y.shape[0])
                # TODO: Implement computation of acc5
                progress_bar.set_postfix(metric_avg.get_all())
 
        print(f"Epoch {epoch} {val_dataset.name}, metrics: {metric_avg.get_all()}")

