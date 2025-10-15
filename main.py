from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import torch

from network import PainterNetwork, ConvolutionalNet

print("Hello, Trajan!")

IMG_FOLDER = "images"
IMG_SIZE = 256

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor()
])

print("Loading dataset...")

dataset = datasets.ImageFolder(IMG_FOLDER, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=8)
test_loader = DataLoader(test_data, batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# input_tensor = torch.input_tensor.to(device)

print("Generating model...")

network = ConvolutionalNet()

model = network.to(device)

print("Model generated:")
print(model)
