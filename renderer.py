import random
import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image
from scipy.ndimage import rotate

# If you're seeing NaNs in the network outputs, it's probably because a stroke was rendered that was so small or so transparent that it did not
# affect a single pixel. That can be fixed by increasing the scale or alpha offset, which causes the stroke to be rendered regardless
SCALE_OFFSET = 0.01
SCALE_SCALING = 0.5
ALPHA_OFFSET = 0.2
ALPHA_SCALING = 0.8

class Renderer():


    def __init__(self, device, height=64, width=64):
        # Precompute meshgrid
        self.device = device
        self.yy, self.xx = torch.meshgrid(
            torch.linspace(0, 1, height, device=device),
            torch.linspace(0, 1, width, device=device),
            indexing='ij'
        )

    def render_stroke(self, stroke_params):
        """
        Vectorized multi-stroke rendering.

        stroke_params: [B, 8] = (x, y, scale, angle, r, g, b, alpha)
        """

        # Stroke params
        x = stroke_params[:, 0]
        y = stroke_params[:, 1]
        scale = SCALE_OFFSET + SCALE_SCALING * stroke_params[:, 2] # stops it from ever being 0, while also not messing with the gradient by clamping
        angle = stroke_params[:, 3] * 2 * torch.pi
        color = stroke_params[:, 4:7] # [B,3]
        alpha = ALPHA_OFFSET + ALPHA_SCALING * stroke_params[:, 7] # stops it from just generating invisible strokes

        x = x[:, None, None]
        y = y[:, None, None]
        scale = scale[:, None, None]
        angle = angle[:, None, None]
        color = color[:, :, None, None]
        alpha = alpha[:, None, None]

        # Elongate in the y direction and shrink in the x direction
        sx = (scale * 0.5)
        sy = (scale * 2.5)

        dx = self.xx - x
        dy = self.yy - y

        # Rotation (currently zero)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        xr = cos_a * dx + sin_a * dy
        yr = -sin_a * dx + cos_a * dy

        # Gaussian mask
        mask = torch.exp(-torch.pow(torch.sqrt(xr**2 / (2*sx**2) + yr**2 / (2*sy**2)), 4))
        mask = mask / mask.amax(dim=(-2, -1), keepdim=True)
        mask = mask * 1.1
        mask = mask * alpha
        mask = torch.unsqueeze(mask, 1)

        # Apply stroke
        self.canvas = self.canvas * (1 - mask) + color * mask

    def initialize_canvas(self, images):
        B, _, _, _ = images.shape
        # Initialize the canvas with the averages of the images used instead of pure white
        means = torch.mean(images, (2,3)).view(B, 3, 1, 1).repeat((1, 1, 64, 64))
        self.canvas = means.to(self.device)

    def initialize_nondiff_canvas(self, ax, image):
        means = torch.mean(image, (2,3)).view(3, 1, 1).repeat((1, 64, 64))
        background = means.detach().cpu().permute(1,2,0).numpy()
        ax.imshow(background, extent=(-32, 32, 32, -32), aspect="equal", origin="lower")


    def render_nondifferentiable(self, ax, stroke_params, stroke_fp="strokes/simple_stroke.png"):
        # First we load the stroke image from the file
        stroke_img = Image.open(stroke_fp).convert("RGBA")
        x = stroke_params[0].item() * 64 - 32
        y = -stroke_params[1].item() * 64 + 32
        scale = SCALE_OFFSET + SCALE_SCALING * (stroke_params[2].item() * 0.4)
        angle = -(stroke_params[3].item() * 360) + 90
        r = stroke_params[4].item() * 255
        g = stroke_params[5].item() * 255
        b = stroke_params[6].item() * 255
        alpha = ALPHA_OFFSET + ALPHA_SCALING * stroke_params[7].item()
        # Then rotate the stroke image by the given angle
        stroke_img = rotate(stroke_img, angle, reshape=True)
        # In order to recolor the stroke image, need to convert it to a numpy array and manually set the rgb values (leaving the alpha value alone for now)
        stroke_img_array = np.asarray(stroke_img)
        stroke_img_array[:, :, 0] = r
        stroke_img_array[:, :, 1] = g
        stroke_img_array[:, :, 2] = b
        stroke_img = Image.fromarray(stroke_img_array, "RGBA")
        # Then we calculate the position and size of the stroke image
        x_size = stroke_img_array.shape[1]
        y_size = stroke_img_array.shape[0]
        x_start = x - (scale * x_size / 2)
        y_start = y - (scale * y_size / 2)
        x_end = x + (scale * x_size / 2)
        y_end = y + (scale * y_size / 2)
        # Finally, show the image. Note that we apply alpha at this step because it's easier
        ax.imshow(stroke_img, extent=(x_start, x_end, y_end, y_start), alpha=alpha, aspect="equal", origin="lower")

# You can run this file to see how the differentiable renderer compares to the non-differentiable one
if __name__ == "__main__":
    num_strokes = 2
    rend = Renderer("cpu")
    while True:
        rend.initialize_canvas(1) # doesn't wrok. sorry
        _, canvas_axs = plt.subplots(1,2,figsize=(8,4))
        canvas_axs[1].set_xlim(-32, 32)
        canvas_axs[1].set_ylim(-32, 32)
        for i in range(num_strokes):
            stroke = torch.tensor([
                random.random(),                    # x
                random.random(),                    # y
                random.random() * 0.1 + 0.1,        # scale
                random.random(),                    # angle
                random.random(),                    # r
                random.random(),                    # g
                random.random(),                    # b
                random.random() * 0.5 + 0.5         # alpha
            ])
            rend.render_nondifferentiable(canvas_axs[1], stroke)
            stroke = torch.unsqueeze(stroke, 0) # simulating a batch size of 1
            rend.render_stroke(stroke)
        canvas_0 = rend.canvas[0].detach().cpu().permute(1,2,0).numpy()
        canvas_axs[0].imshow(canvas_0)
        plt.show()
        plt.close("all")