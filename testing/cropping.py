import argparse
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import torch
import os
import time
from tqdm import tqdm

import sys
sys.path.append('/root/CS570-Final-Project/src/cropping')

from crop import Cropper
from classify_crop import ImageClassifier

def main(image_dir, method, num_crops, verbose):

    # List to store all image paths
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, img))]

    # Instantiate the cropper and classifier
    cropper = Cropper(method=method, num_crops=num_crops)
    classifier = ImageClassifier()

    for image_path in tqdm(image_paths, desc="Processing images"):
        start_time = time.time()

        # Load the image
        image = Image.open(image_path)

        time_load = time.time() - start_time
        start_time = time.time()

        # Crop the image
        cropped_images = cropper.crop(image)

        time_crop = time.time() - start_time
        start_time = time.time()

        # Classify the cropped images
        predictions = classifier.classify(cropped_images)

        time_classify = time.time() - start_time

        if verbose:
            print(f"Time taken to load  images: {time_load} seconds")
            print(f"Time taken to crop  images: {time_crop} seconds")
            print(f"Time taken to classify images: {time_classify} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('--image_dir', default="/root/dataset/val", help='Path to image directory')
    parser.add_argument('--method', default='random', choices=['random', 'rpn'], help='Cropping method')
    parser.add_argument('--num_crops', type=int, default=5, help='Number of crops (for random method)')
    parser.add_argument('--verbose', type=int, default=0, choices=[0, 1], help='Whether to print verbose output')

    args = parser.parse_args()

    main(args.image_dir, args.method, args.num_crops, args.verbose)
