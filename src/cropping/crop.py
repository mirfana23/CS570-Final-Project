import torchvision.transforms as transforms
import torchvision
import torch
from PIL import Image
import random

class Cropper:
    def __init__(self, method='random', output_size=(224, 224), crop_size=(224, 224), crop_size_method='fixed', num_crops=1, resize_rpn_output=True, device='cuda:0'):
        """
        Initialize the Cropper class.

        Args:
        - method (str): The method to use for cropping. Can be 'random' or 'rpn' (Region Proposal Network).
        - output_size (tuple): The size of the output image after cropping.
        - crop_size (tuple): The size of the cropping region.
        - crop_size_method (str): The method to use for selecting crop size. Can be 'fixed' or 'range'.
        - num_crops (int): The number of crops to produce when using 'random' method.
        - resize_rpn_output (bool): Whether to resize the output of the RPN to output_size.
        """

        self.method = method
        self.output_size = output_size
        self.crop_size = crop_size
        self.crop_size_method = crop_size_method
        self.num_crops = num_crops
        self.resize_rpn_output = resize_rpn_output
        self.device = device

        if self.method == 'rpn':
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
            self.model = self.model.to(device).eval()  # Move model to GPU

        self.transform = transforms.Resize(self.output_size, antialias=True)

    def random_crop(self, image):
        """
        Perform random cropping on the input image.
        """

        B, channels, height, width = image.shape
        output_images = []

        for _ in range(self.num_crops):
            if self.crop_size_method == 'fixed':
                # Adjust crop size if it's larger than the image size 
                if self.crop_size[0] >= width or self.crop_size[1] >= height:
                    self.crop_size = (int(width * 0.9), int(height * 0.9))

                crop_width, crop_height = self.crop_size

            elif self.crop_size_method == 'range':
                min_size = max(100, min(width, height))
                max_size = min(224, width, height)

                # If the image size is less than 100, adjust the size to be 90% of the image size
                if min_size == min(width, height) and min_size < 100:
                    min_size = max_size = int(min_size * 0.9)

                min_size = min(min_size, max_size)

                crop_width = crop_height = random.randint(min_size, max_size + 1)  # Add +1 to handle edge cases
                
            else:
                raise ValueError(f"Invalid crop size method: {self.crop_size_method}")

            max_x = max(0, width - crop_width)
            max_y = max(0, height - crop_height)

            # If the crop size is equal to the image size, adjust the crop size
            if max_x == max_y == 0:
                crop_width = int(width * 0.9)
                crop_height = int(height * 0.9)
                max_x = max(0, width - crop_width)
                max_y = max(0, height - crop_height)

            x = torch.randint(0, max_x + 1, (1,)).item()
            y = torch.randint(0, max_y + 1, (1,)).item()

            cropped_image = image[..., y:y+crop_height, x:x+crop_width]

            if (crop_width, crop_height) != self.output_size:
                cropped_image = self.transform(cropped_image)

            output_images.append(cropped_image)

        return output_images

    def rpn_crop(self, img_tensor):
        """
        Perform cropping using a Region Proposal Network (RPN) on the input image.
        """

        # Move the tensor to the same device as the model
        img_tensor = img_tensor.to(self.device)
        output_images = []

        for img in img_tensor:
            # Add an extra dimension
            img = img.unsqueeze(0)

            # Get the model's predictions
            with torch.no_grad():
                prediction = self.model(img)

            # Extract bounding boxes from model prediction
            boxes = prediction[0]['boxes']

            for box in boxes:
                # Convert tensor to a tuple of integers
                box = tuple(map(int, box.round().tolist()))

                # Crop the image using the bounding box coordinates
                cropped_image = img[0, ..., box[1]:box[3], box[0]:box[2]]

                # Resize the cropped image if necessary
                if self.resize_rpn_output:
                    cropped_image = self.transform(cropped_image)

                # Apply transformations and add to the list of output images
                output_images.append(cropped_image)

        return output_images

    def crop(self, image):
        """
        Perform cropping on the input image using the specified method.
        """

        if self.method == 'random':
            return self.random_crop(image)
        elif self.method == 'rpn':
            return self.rpn_crop(image)
        else:
            raise ValueError(f"Invalid cropping method: {self.method}")

