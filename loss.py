import torch
import torch.nn as nn
import torchvision.models as models
    
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

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
    

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True