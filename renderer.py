import random
import sys
import csv
import os

import matplotlib.pyplot as plt
# import torch.nn as nn
import numpy as np
import pandas as pd

from PIL import Image
from scipy.ndimage import rotate
from torchvision import datasets
from torchvision.io import decode_image
from torch.utils.data import Dataset

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



# class DifferentiableRenderer(nn.Module):
#     def __init__(self):
#         super().__init__()

class StrokeDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "generate":
        num_images_to_generate = int(sys.argv[2])
        with open("images/stroke_dataset/annotations.csv", 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=' ', quotechar="|", quoting=csv.QUOTE_MINIMAL)
            for i in range(num_images_to_generate):
                # Set up the plot with the right size and some random pixels
                print(i/num_images_to_generate)
                plt.xlim(-32, 32)
                plt.ylim(-32, 32)
                plt.axis("off")
                plt.imshow(np.random.rand(64, 64, 3), extent=(-32, 32, -32, 32))
                plt.savefig("images/stroke_dataset/before/" + str(i) + ".jpg", bbox_inches='tight', pad_inches=0.0)
                # Generate random stroke parameters
                x = random.randint(-32, 32)
                y = random.randint(-32, 32)
                angle = random.randint(0, 360)
                scale = random.random() * 0.1
                r = random.random() * 255
                g = random.random() * 255
                b = random.random() * 255
                alpha = random.random()
                # Make sure we store the parameters so the model can learn they mean
                csv_writer.writerow([str(i)+".png", x, y, angle, scale, r, g, b, alpha])
                # Render it to a figure and save it as a piece of training data
                render_nondifferentiable(plt, x, y, angle, scale, r, g, b, alpha)
                plt.savefig("images/stroke_dataset/after/" + str(i) + ".jpg", bbox_inches='tight', pad_inches=0.0)
    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        dataset = datasets.ImageFolder("images/stroke_dataset")
    else:
        print("Usage: \"renderer.py generate n\" to generate n training data images or \"renderer.py train\" to train the differentiable renderer")