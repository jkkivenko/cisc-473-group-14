import random
import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image
from scipy.ndimage import rotate

# If you're seeing NaNs in the network outputs, it's probably because a stroke was rendered that was so small or so transparent that it did not
# affect a single pixel. That can be fixed by increasing the scale or alpha offset, which causes the stroke to be rendered regardless
ALPHA_OFFSET = 0.2
ALPHA_SCALING = 0.8
STROKE_SQUISHING_FACTOR = 2

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

        stroke_params: [B, 8] = (x1, y1, x2, y2, scale, r, g, b, alpha)
        """

        # Stroke params
        x1 = stroke_params[:, 0]
        y1 = stroke_params[:, 1]
        x2 = stroke_params[:, 2]
        y2 = stroke_params[:, 3]
        color = stroke_params[:, 4:7] # [B,3]
        alpha = ALPHA_OFFSET + ALPHA_SCALING * stroke_params[:, 7] # stops it from just generating invisible strokes

        x1 = x1[:, None, None]
        y1 = y1[:, None, None]
        x2 = x2[:, None, None]
        y2 = y2[:, None, None]
        color = color[:, :, None, None]
        alpha = alpha[:, None, None]

        dx1 = self.xx - x1
        dy1 = self.yy - y1
        dx2 = self.xx - x2
        dy2 = self.yy - y2
        
        # I hate math
        radius = torch.sqrt(torch.square(x2-x1) + torch.square(y2-y1))
        ellipse_distance = torch.sqrt(torch.square(dx1) + torch.square(dy1)) + torch.sqrt(torch.square(dx2) + torch.square(dy2)) - radius
        ellipse_distance *= STROKE_SQUISHING_FACTOR
        
        sigma = 0.2
            
        # Gaussian mask
        mask = torch.exp(-torch.square(ellipse_distance / sigma))
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
        x1 = stroke_params[0].item() * 64 - 32
        y1 = -stroke_params[1].item() * 64 + 32
        x2 = stroke_params[2].item() * 64 - 32
        y2 = -stroke_params[3].item() * 64 + 32
        r = stroke_params[4].item() * 255
        g = stroke_params[5].item() * 255
        b = stroke_params[6].item() * 255
        alpha = ALPHA_OFFSET + ALPHA_SCALING * stroke_params[7].item()
        # Calculate center
        x = (x1 + x2) / 2.0
        y = (y1 + y2) / 2.0

        scale = np.sqrt((x2-x1)**2 + (y2-y1)**2) / 750
        angle = np.atan((y2-y1)/(x2-x1)) * 360 / (2 * torch.pi)

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
    num_strokes = 4
    rend = Renderer("cpu")
    while True:
        blank_image = torch.ones((1, 3, 64, 64))
        rend.initialize_canvas(blank_image)
        _, canvas_axs = plt.subplots(1,2,figsize=(8,4))
        canvas_axs[1].set_xlim(-32, 32)
        canvas_axs[1].set_ylim(-32, 32)
        for i in range(num_strokes):
            stroke = torch.tensor([
                random.random(),                    # x1
                random.random(),                    # y1
                random.random(),                    # x2
                random.random(),                    # y2
                random.random(),                    # r
                random.random(),                    # g
                random.random(),                    # b
                random.random()                     # alpha
            ])
            print(stroke)
            rend.render_nondifferentiable(canvas_axs[1], stroke)
            stroke = torch.unsqueeze(stroke, 0) # simulating a batch size of 1
            rend.render_stroke(stroke)
        canvas_0 = rend.canvas[0].detach().cpu().permute(1,2,0).numpy()
        canvas_axs[0].imshow(canvas_0)
        plt.show()
        plt.close("all")