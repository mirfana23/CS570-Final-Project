from PIL import Image
import torchvision.transforms as transforms
import torchvision
import torch

class Cropper:
    def __init__(self, method='random', output_size=(224, 224), num_crops=1):
        self.method = method
        self.output_size = output_size
        self.num_crops = num_crops

        if self.method == 'rpn':
            # Load pre-trained Faster R-CNN with Resnet50 backbone
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            self.model = self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(self.output_size),
            transforms.ToTensor()
        ])

    def random_crop(self, image):
        width, height = image.size
        output_images = []

        for _ in range(self.num_crops):
            max_x = width - self.output_size[0]
            max_y = height - self.output_size[1]

            x = torch.randint(0, max_x, (1,)).item()
            y = torch.randint(0, max_y, (1,)).item()

            cropped_image = image.crop((x, y, x + self.output_size[0], y + self.output_size[1]))
            output_images.append(self.transform(cropped_image))
        
        return output_images

    def rpn_crop(self, image):
        # Convert image to tensor
        img_tensor = self.transform(image)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            prediction = self.model(img_tensor)

        # Extract bounding boxes from model prediction
        boxes = prediction[0]['boxes']

        output_images = []
        for box in boxes:
            cropped_image = image.crop(box)
            output_images.append(self.transform(cropped_image.resize(self.output_size)))

        return output_images

    def crop(self, image):
        if self.method == 'random':
            return self.random_crop(image)
        elif self.method == 'rpn':
            return self.rpn_crop(image)
        else:
            raise ValueError(f"Invalid cropping method: {self.method}")
