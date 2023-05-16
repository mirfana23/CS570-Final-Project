import os
from torch.utils.data import Dataset
from PIL import Image
import json
import torch
import torchvision.transforms as transforms

class ImgNetDataset(Dataset):
    """
    Dataset for ImageNet
    json files will be in the form of:
    {
        "data": [
            {
                "img_path": "path/to/image", // absolute directory
                "labels": [1, 2, 3, 4, 5] // list of label indices (check: /root/CS570-Final-Project/datasets/imagenet_class_index.json)
            },
            ...
        ]
    }
    """
    def __init__(self, json_path, type='train'):
        with open(json_path, "rb") as f:
            self.data = json.load(f)['data']
        self.type = type
        if self.type == 'train':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(224, antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]['img_path']
        x = Image.open(img_path).convert("RGB")
        x = self.transform(x)

        labels = torch.tensor(self.data[idx]['labels'])
        y = torch.zeros(1000)
        if len(labels) > 0:
            y[labels] = 1
        
        return x, y
