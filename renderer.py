import random
import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image
from scipy.ndimage import rotate

# This is an example of a non-differentiable renderer. We use this to do two things:
# 1. To generate training data for the differentiable renderer
# 2. To render the strokes at inference time because they look nicer than the non-differentiable renderer
def render_nondifferentiable(canvas, x, y, angle, scale, r, g, b, alpha, stroke_fp="strokes/simple_stroke.png"):
    # First we load the stroke image from the file
    img = Image.open(stroke_fp).convert("RGBA")
    # Then rotate the stroke image by the given angle
    img = rotate(img, angle, reshape=True)
    # In order to recolor the stroke image, need to convert it to a numpy array and manually set the rgb values (leaving the alpha value alone for now)
    img_array = np.asarray(img)
    img_array[:, :, 0] = r
    img_array[:, :, 1] = g
    img_array[:, :, 2] = b
    img = Image.fromarray(img_array, "RGBA")
    # Then we calculate the position of the stroke image
    x_size = img_array.shape[1]
    y_size = img_array.shape[0]
    x_start = x - (scale * x_size / 2)
    y_start = y - (scale * y_size / 2)
    x_end = x + (scale * x_size / 2)
    y_end = y + (scale * y_size / 2)
    # Finally, show the image. Note that we apply alpha at this step because it's easier
    canvas.imshow(img, extent=(x_start, x_end, y_end, y_start), alpha=alpha, aspect="equal", origin="lower")

# -------------------------------
# Stroke rendering
# -------------------------------
def render_differentiable(canvas, x, y, scale, angle, color, alpha):
    """
    Draw a slightly elongated, rotated stroke on the canvas.
    
    canvas: [B, C, H, W]
    x, y: [B] normalized center coordinates [0,1]
    scale: [B] base scale (used for both axes, elongation added internally)
    color: [B, C] stroke color
    """
    B, C, H, W = canvas.shape
    device = canvas.device

    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, H, device=device),
        torch.linspace(0, 1, W, device=device),
        indexing='ij'
    )
    xx = xx.unsqueeze(0).expand(B, -1, -1)  # [B,H,W]
    yy = yy.unsqueeze(0).expand(B, -1, -1)

    # shift coordinates to center
    dx = xx - x[:, None, None]
    dy = yy - y[:, None, None]

    # calculate stroke rotation
    angle.to(device)
    cos_a = torch.cos(angle)[:, None]
    sin_a = torch.sin(angle)[:, None]

    # rotate coordinates
    xr = cos_a * dx + sin_a * dy
    yr = -sin_a * dx + cos_a * dy

    # make ellipse: 0.5*scale for x, 1.2*scale for y (elongation)
    sx = (0.5 * scale).clamp_min(0.05)[:,None, None]
    sy = (1.2 * scale).clamp_min(0.05)[:,None, None]

    mask = torch.exp(-torch.pow(torch.sqrt(xr**2 / (2*sx**2) + yr**2 / (2*sy**2)), 4))

    # normalize and scale opacity
    mask = mask / mask.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    mask = mask * 1.1
    mask = mask * alpha[:,None]
    mask = mask.unsqueeze(1)  # [B,1,H,W]

    color = color[:, :, None, None]  # [B,C,1,1]

    new_canvas = canvas * (1 - mask) + color * mask
    return new_canvas


# You can run this file to see how the differentiable renderer compares to the non-differentiable one
if __name__ == "__main__":
    while True:
        x = random.random()
        y = random.random()
        scale = random.random() * 0.8 + 0.2
        angle = random.random() * 2 * torch.pi
        r = random.random()
        g = random.random()
        b = random.random()
        alpha = random.random() * 0.5 + 0.5

        _, canvas_axs = plt.subplots(1,2,figsize=(8,4))
        color = torch.tensor([[r,g,b]])
        canvas_0 = render_differentiable(torch.ones((1, 3, 64, 64)), torch.tensor([x]), torch.tensor([y]), torch.tensor([scale]), torch.tensor([angle]), color, torch.tensor([alpha]))
        canvas_0 = canvas_0[0].detach().cpu().permute(1,2,0).numpy()
        canvas_axs[0].imshow(canvas_0)
        canvas_axs[1].set_xlim(-32, 32)
        canvas_axs[1].set_ylim(-32, 32)
        render_nondifferentiable(canvas_axs[1], x * 64 - 32, -y * 64 + 32, -(360 * angle / (2*torch.pi)) - 90, scale / 4, r * 255, g * 255, b * 255, alpha)
        plt.show()