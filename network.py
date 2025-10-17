import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Define model
class PainterNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    # NOTE: this shoylkd be convolution

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class ConvolutionalNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

class NeuralPainter(nn.Module):
    def __init__(self, stroke_dim=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, stroke_dim * 5)
        )

    def forward(self, target_img):
        features = self.conv(target_img).view(target_img.size(0), -1)
        stroke_params = torch.sigmoid(self.fc(features))
        return stroke_params
    
class PerceptualLoss(nn.Module):
    """
    Computes a perceptual (feature-space) loss between two images using
    pretrained VGG16 feature maps.

    Parameters
    ----------
    layer_ids : list[int]
        Indices of VGG16 layers whose outputs will be compared.
        Lower indices -> low-level edges/colors, higher -> textures/objects.
    weights : list[float] | None
        Optional relative weights per layer (same length as layer_ids).
    """
    def __init__(self, layer_ids=[3, 8, 15, 22], weights=None):
        super().__init__()
        vgg_pretrained = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features
        layers = []
        for layer in vgg_pretrained:
            # Replace in-place ReLUs with out-of-place versions
            if isinstance(layer, nn.ReLU):
                layers.append(nn.ReLU(inplace=False))
            else:
                layers.append(layer)
        vgg = nn.Sequential(*layers)
        self.layers = nn.ModuleList()
        start = 0
        for end in layer_ids:
            self.layers.append(nn.Sequential(*[vgg[i] for i in range(start, end)]))
            start = end
        for p in self.parameters():
            p.requires_grad = False  # freeze pretrained VGG
        self.weights = weights or [1.0] * len(self.layers)

    def forward(self, input, target):
        """
        Both input and target must be normalized to [0,1] range.
        Automatically applies ImageNet mean/std normalization.
        """
        # Normalize for VGG
        mean = torch.tensor([0.485, 0.456, 0.406], device=input.device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=input.device).view(1,3,1,1)
        input_norm = (input - mean) / std
        target_norm = (target - mean) / std

        loss = 0.0
        for w, layer in zip(self.weights, self.layers):
            input_norm = layer(input_norm)
            target_norm = layer(target_norm)
            loss += w * nn.functional.mse_loss(input_norm, target_norm)
        return loss