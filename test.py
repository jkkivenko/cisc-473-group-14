import random
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.fid import FrechetInceptionDistance as FID

from loss import EarlyStopping, PerceptualLoss, stroke_loss
from network import NeuralPainter
from renderer import Renderer

IMG_SIZE = 64
BATCH_SIZE = 12
DISPLAY_EVERY_N_EPOCHS = 1
STROKE_LOSS_AMOUNT = 0.8
PIXEL_LOSS_AMOUNT = 3000
PERCEPTUAL_LOSS_AMOUNT = 0.1

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    if len(sys.argv) > 3:
        model_filepath = sys.argv[1]
        img_folder_filepath = sys.argv[2]
        num_strokes = int(sys.argv[3])
        with open(model_filepath, "rb") as f:
            dataset = datasets.ImageFolder(img_folder_filepath, transform=transform)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            model = torch.load(f, weights_only=False, map_location=device).to(device)

            for (image, _) in loader:

                fig, canvas_axs = plt.subplots(1,3,figsize=(10,4))
                canvas_axs[0].set_title("Target")
                canvas_axs[0].axis('off')
                canvas_axs[1].set_title("Model View")
                canvas_axs[1].axis('off')

                canvas_axs[2].clear()
                canvas_axs[2].set_title("Rendered Output")
                canvas_axs[2].axis('off')
                canvas_axs[2].set_xlim(-32, 32)
                canvas_axs[2].set_ylim(-32, 32)

                canvas_axs[0].imshow(image[0].permute(1,2,0).numpy())

                rend = Renderer(device)
                rend.initialize_canvas(image)
                rend.initialize_nondiff_canvas(canvas_axs[2], image)
                # Predict multiple strokes per image
                for stroke_idx in range(num_strokes):
                    stroke_params = model(torch.cat([image, rend.canvas], dim=1))
                    rend.render_stroke(stroke_params)
                    rend.render_nondifferentiable(canvas_axs[2], torch.squeeze(stroke_params))
                canv = rend.canvas[0].detach().cpu().permute(1,2,0).numpy()
                canvas_axs[1].imshow(canv)
                plt.show(block=True)
    else:
        print("Usage: python3 test.py MODEL_FILEPATH IMAGE_FOLDER_FILEPATH NUM_STROKES")