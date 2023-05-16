import torchvision.transforms as transforms
import torchvision
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Normalize, Resize, CenterCrop, InterpolationMode


import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Normalize, Resize, CenterCrop, InterpolationMode, Compose
import torchvision.transforms as transforms

class ImageClassifier:
    def __init__(self, device='cuda:0'):
        # Load pre-trained ResNet50 model
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model = self.model.to(device).eval()
        self.device = device

        # Create transform to normalize input image
        self.transforms = Compose([
            Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=True),
            CenterCrop(224),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def classify(self, image_tensors):
        # Convert list of tensors to a single batch tensor
        image_batch = torch.stack(image_tensors).to(self.device)

        # Normalize tensors
        img_batch = self.transforms(image_batch)

        with torch.no_grad():
            # Perform inference
            output = self.model(img_batch)

            # Convert output probabilities to predicted class
            _, preds = torch.max(output, 1)

        return preds.tolist()
