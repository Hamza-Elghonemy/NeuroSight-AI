import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image

CLASSES = ['hemorrhagic', 'ischemic', 'tumor']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class CBM(nn.Module):
    def __init__(self, num_concepts=10, num_classes=3):
        super().__init__()
        self.backbone = models.resnet50(pretrained=False)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_concepts)
        self.c2y = nn.Linear(num_concepts, num_classes)
        
    def forward(self, x):
        c_logits = self.backbone(x)
        c_probs = torch.sigmoid(c_logits)
        y_logits = self.c2y(c_probs)
        return y_logits, c_probs

def load_model(model_type='cnn', model_path=None):
    if model_type == 'cbm':
        model = CBM(num_classes=len(CLASSES))
    elif model_type == 'unet':
        model = UNet()
    else:
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    
    if model_path:
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"Loaded {model_type} from {model_path}")
        except FileNotFoundError:
            print(f"Warning: Model file {model_path} not found. Using random weights.")
            
    model = model.to(DEVICE)
    model.eval()
    return model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        
        if isinstance(output, tuple):
            output = output[0]
            
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
            
        score = output[:, class_idx].squeeze()
        score.backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = torch.mean(gradients, dim=(1, 2)).view(-1, 1, 1)
        cam = torch.sum(weights * activations, dim=0)
        cam = F.relu(cam)
        
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, CLASSES[class_idx]

# --- U-Net Architecture ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(3, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x = self.up1(x3)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up2(x)
        
        return self.outc(x)
