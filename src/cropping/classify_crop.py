import torchvision.transforms as transforms
import torchvision
import torch

from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
from torchvision.transforms import Normalize, Resize, CenterCrop, InterpolationMode, Compose

class ImageClassifier:
    def __init__(self, model=None, device='cuda:0'):
        if model is None:
            # Load pre-trained ResNet50 model if not specified
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            
        self.model = model.to(device).eval()
        self.device = device

        self.transforms = transforms.Compose([
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def classify(self, image_tensors):

        # Normalize and process each image tensor separately
        processed_images = []
        for img_tensor in image_tensors:
            img_tensor = self.transforms(img_tensor.squeeze(0)) # apply transforms
            processed_images.append(img_tensor.unsqueeze(0))

        # If no images were processed, return a list of 1000 zeros
        if not processed_images:
            return [0]*1000

        # Convert list of tensors to a single batch tensor
        image_batch = torch.cat(processed_images).to(self.device)


        with torch.no_grad():
            # Perform inference
            output = self.model(image_batch)

            # Convert output probabilities to predicted class
            _, preds = torch.max(output, 1)

            # Create one-hot encoding for each predicted class
            one_hot_preds = torch.zeros((preds.shape[0], 1000), device=self.device)
            one_hot_preds[range(preds.shape[0]), preds] = 1
            one_hot_preds = torch.any(one_hot_preds, dim=0).int().tolist() # perform logical OR along the batch dimension

        return one_hot_preds
