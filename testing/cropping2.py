import argparse
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torchvision
import torch
import os
import time
from tqdm import tqdm
from torchvision.io import read_image

import sys
# Add the project root directory to the sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cropping.crop import Cropper
from src.cropping.classify_crop import ImageClassifier

from src.cropping.crop import Cropper
from src.cropping.classify_crop import ImageClassifier
from src.dataloaders.dataloader import ImgNetDataset, public_tr

def main(json_dir, method, num_crops, verbose):

    # List to store all image paths
    dataset = ImgNetDataset(json_path=json_dir, type='val')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Instantiate the cropper and classifier
    cropper = Cropper(method=method, num_crops=num_crops)
    classifier = ImageClassifier()
    start_time = time.time()
    for img, labels in tqdm(dataloader, desc="Processing images"):
        end_time = time.time()
        time_load = end_time - start_time
        start_time = time.time()

        # Crop the image
        cropped_images = cropper.crop(img)

        time_crop = time.time() - start_time
        start_time = time.time()

        # Classify the cropped images
        predictions = classifier.classify(cropped_images)

        time_classify = time.time() - start_time

        if verbose:
            print(f"Time taken to load  images: {time_load} seconds")
            print(f"Time taken to crop  images: {time_crop} seconds")
            print(f"Time taken to classify images: {time_classify} seconds")
        start_time = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('--json_dir', default="/root/CS570-Final-Project/datasets/imgnet1k_original/val.json", help='Path to image directory')
    parser.add_argument('--method', default='random', choices=['random', 'rpn'], help='Cropping method')
    parser.add_argument('--num_crops', type=int, default=5, help='Number of crops (for random method)')
    parser.add_argument('--verbose', type=int, default=0, choices=[0, 1], help='Whether to print verbose output')

    args = parser.parse_args()

    main(args.json_dir, args.method, args.num_crops, args.verbose)
