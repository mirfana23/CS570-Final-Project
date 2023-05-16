import torchvision.transforms as transforms
import torchvision
import torch
from PIL import Image

class Cropper:
    def __init__(self, method='random', output_size=(224, 224), crop_size=(224, 224), num_crops=1, resize_rpn_output=True, device='cuda:0'):
        """
        Initialize the Cropper class.

        Args:
        - method (str): The method to use for cropping. Can be 'random' or 'rpn' (Region Proposal Network).
        - output_size (tuple): The size of the output image after cropping.
        - crop_size (tuple): The size of the cropping region.
        - num_crops (int): The number of crops to produce when using 'random' method.
        - resize_rpn_output (bool): Whether to resize the output of the RPN to output_size.
        """

        self.method = method
        self.output_size = output_size
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.resize_rpn_output = resize_rpn_output
        self.device = device

        # If we're using the 'rpn' method, load a pre-trained Faster R-CNN model.
        if self.method == 'rpn':
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
            self.model = self.model.to(device).eval()  # Move model to GPU

        # Define the image transformations: resizing and converting to tensor
        self.transform = transforms.Compose([
            transforms.Resize(self.output_size),
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.to(self.device))  # Move tensor to the specified device
        ])

    def random_crop(self, image):
        """
        Perform random cropping on the input image.
        """

        width, height = image.size

        # Adjust crop size if it's larger than the image size 
        if self.crop_size[0] >= width or self.crop_size[1] >= height:
            self.crop_size = (int(width * 0.9), int(height * 0.9))
          
        output_images = []

        # Produce the required number of crops
        for _ in range(self.num_crops):
            
            # Compute the maximum x and y coordinates for the top-left corner of the crop
            max_x = width - self.crop_size[0]
            max_y = height - self.crop_size[1]

            # Randomly select the top-left corner of the crop
            x = torch.randint(0, max_x, (1,)).item()
            y = torch.randint(0, max_y, (1,)).item()

            # Crop the image and resize if necessary
            cropped_image = image.crop((x, y, x + self.crop_size[0], y + self.crop_size[1]))
            if self.crop_size != self.output_size:
                cropped_image = cropped_image.resize(self.output_size)
            
            # Apply transformations and add to the list of output images
            output_images.append(self.transform(cropped_image))
        
        return output_images

    def rpn_crop(self, image):
        """
        Perform cropping using a Region Proposal Network (RPN) on the input image.
        """

        # Move the image to GPU and add batch dimension
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Get the model's predictions
        with torch.no_grad():
            prediction = self.model(img_tensor)

        # Extract bounding boxes from model prediction
        boxes = prediction[0]['boxes']

        output_images = []
        for box in boxes:
            # Convert tensor to a tuple of integers
            box = tuple(map(int, box.round().tolist()))

            # Crop the image using the bounding box coordinates
            cropped_image = image.crop(box)

            # Resize the cropped image if necessary
            if self.resize_rpn_output:
                cropped_image = cropped_image.resize(self.output_size)

            # Apply transformations and add to the list of output images
            output_images.append(transforms.ToTensor()(cropped_image))

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

