import torch
import torch.nn as nn

# -------------------------------
# Neural Painter Network (multiple strokes)
# -------------------------------
class NeuralPainter(nn.Module):
    def __init__(self, strokes_per_step=5):
        super().__init__()
        self.strokes_per_step = strokes_per_step

        # Conv backbone: input = target+canvas (6 channels)
        self.conv = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # FC: output = 6 parameters per stroke * strokes_per_step
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 6 * strokes_per_step)
        )

    def forward(self, target_canvas):
        B = target_canvas.size(0)
        features = self.conv(target_canvas).view(B, -1)
        stroke_params = self.fc(features)  # [B, 6 * N]
        stroke_params = torch.sigmoid(stroke_params)  # keep values in [0,1]
        return stroke_params