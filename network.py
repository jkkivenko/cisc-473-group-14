import torch
import torch.nn as nn
import torchvision

# -------------------------------
# Neural Painter Network (multiple strokes)
# -------------------------------
class NeuralPainter(nn.Module):
    def __init__(self, stroke_size=8):
        super().__init__()

        self.nathan = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        
        out_channels = self.nathan.conv1.out_channels
        self.nathan.conv1 = nn.Conv2d(6, out_channels, 7, stride=2)

        num_features = self.nathan.fc.in_features
        self.nathan.fc = nn.Linear(num_features, stroke_size)

        # # Conv backbone: input = target+canvas (6 channels)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(6, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, 3, padding=1),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d(1)
        # )

        # # FC: output = 8 parameters per stroke
        # self.fc = nn.Sequential(
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, stroke_size)
        # )

    def forward(self, target_canvas):
        B = target_canvas.size(0)
        # features = self.conv(target_canvas).view(B, -1)
        # stroke_params = self.fc(features)  # [B, 6 * N]
        stroke_params = self.nathan(target_canvas)
        stroke_params = torch.sigmoid(stroke_params)  # keep values in [0,1]

        # stroke_params[:,3] += torch.ones_like(stroke_params[:,3]) * 0.001 # Do not let scale be zero or else everything breaks

        return stroke_params