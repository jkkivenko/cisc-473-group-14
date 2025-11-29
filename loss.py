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

def stroke_loss(strokes):
    def distance(x1, y1, x2, y2):
        return torch.sqrt(torch.square(x2-x1) + torch.square(y2-y1))

    # strokes is an array of tensors of size B x stroke_size
    # So we iterate over every stroke and compare it to every other stroke
    loss = torch.zeros((strokes[0].shape[0])) # B
    for i in range(len(strokes)):
        stroke_x1 = strokes[i][:, 0] # Bx1
        stroke_y1 = strokes[i][:, 1] # Bx1
        stroke_x2 = strokes[i][:, 2] # Bx1
        stroke_y2 = strokes[i][:, 3] # Bx1
        for j in range(i):
            other_stroke_x1 = strokes[j][:, 0] # Bx1
            other_stroke_y1 = strokes[j][:, 1] # Bx1
            other_stroke_x2 = strokes[j][:, 2] # Bx1
            other_stroke_y2 = strokes[j][:, 3] # Bx1

            # For every comparison, we calculate the distance between the strokes by comparing their p1 and p2 coordinates
            # We have to do it twice to account for the case where a stroke is facing the opposite direction
            # I didn't do a great job of explaining that - see this desmos graph for a more intuitive understanding
            # https://www.desmos.com/calculator/5yzykiijwx
            # By the way, both d1 and d2 are of size Bx1
            
            d1 = distance(stroke_x1, stroke_y1, other_stroke_x2, other_stroke_y2) + distance(stroke_x2, stroke_y2, other_stroke_x1, other_stroke_y1)
            d2 = distance(stroke_x1, stroke_y1, other_stroke_x1, other_stroke_y1) + distance(stroke_x2, stroke_y2, other_stroke_x2, other_stroke_y2)

            loss += 1.0 / (torch.minimum(d1, d2) + 0.01) # plus epsilon so no division by zero
    return torch.sum(loss)
