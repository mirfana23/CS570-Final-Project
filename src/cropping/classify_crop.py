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

    def classify(self, image_tensors):

        # Convert list of tensors to a single batch tensor
        image_batch = torch.cat(image_tensors).to(self.device)


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
