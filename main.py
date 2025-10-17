from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from network import NeuralPainter, PainterNetwork, ConvolutionalNet, PerceptualLoss
import matplotlib.pyplot as plt



print("Hello, Trajan!")

IMG_FOLDER = "images/all_images"
IMG_SIZE = 64

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor()
])

print("Loading dataset...")

dataset = datasets.ImageFolder(IMG_FOLDER, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=1)
test_loader = DataLoader(test_data, batch_size=1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset and loader
transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
dataset = datasets.ImageFolder("images/all_images", transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# Model, loss, optimizer
model = NeuralPainter().to(device)
loss_fn = nn.MSELoss()
perceptual_loss = PerceptualLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def render_stroke(canvas, x, y, radius, color):
    B, C, H, W = canvas.shape
    device = canvas.device
    
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, H, device=device),
        torch.linspace(0, 1, W, device=device),
        indexing='ij'
    )
    xx = xx.unsqueeze(0).expand(B, -1, -1)
    yy = yy.unsqueeze(0).expand(B, -1, -1)

    # Clamp radius for visibility
    radius = radius.clamp(0.05, 0.5)
    
    # Convert normalized radius to fraction of canvas (for Gaussian mask)
    dist = torch.sqrt((xx - x[:, None, None])**2 + (yy - y[:, None, None])**2)
    mask = torch.exp(-(dist**2) / (2 * radius[:, None, None]**2))
    mask = mask.unsqueeze(1)
    
    color = color[:, :, None, None]
    
    # Boost stroke intensity slightly for early training
    mask = mask * 1.5
    mask = mask.clamp(0,1)
    
    new_canvas = canvas * (1 - mask) + color * mask
    return new_canvas

# -------------------------------
# Example Training Loop
# -------------------------------
def train_painter_multi_stroke(model, train_loader, num_epochs, optimizer, loss_fn,
                               perceptual_loss, num_strokes=50, strokes_per_step=5,
                               device='cuda', stroke_noise=True):
    """
    Trains a neural painter that predicts multiple strokes per step.
    
    Parameters:
        strokes_per_step : int
            Number of strokes the network predicts per forward pass.
    """
    model.to(device)

    for epoch in range(num_epochs):
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            canvas = 0.5 * torch.ones_like(images).to(device)  # start gray canvas

            optimizer.zero_grad()

            for step in range(num_strokes):
                # Predict multiple strokes at once
                stroke_params = model(torch.cat([images, canvas], dim=1))  # [B, 6*strokes_per_step]
                B = images.size(0)
                stroke_params = stroke_params.view(B, strokes_per_step, 6)  # [B, N, 6]

                for n in range(strokes_per_step):
                    x, y, radius, r, g, b = stroke_params[:, n, :].split(1, dim=1)

                    # Add small random noise for diversity
                    if stroke_noise:
                        x = x + 0.05 * torch.randn_like(x)
                        y = y + 0.05 * torch.randn_like(y)
                        radius = radius + 0.03 * torch.randn_like(radius)
                        r = r + 0.1 * torch.randn_like(r)
                        g = g + 0.1 * torch.randn_like(g)
                        b = b + 0.1 * torch.randn_like(b)

                    # Clamp values to valid ranges
                    x = x.clamp(0,1)
                    y = y.clamp(0,1)
                    radius = radius.clamp(0.01,0.2)
                    r = r.clamp(0,1)
                    g = g.clamp(0,1)
                    b = b.clamp(0,1)

                    color = torch.cat([r,g,b], dim=1)

                    # Apply stroke to canvas
                    canvas = render_stroke(canvas, x.squeeze(1), y.squeeze(1), radius.squeeze(1), color)

            # Compute loss
            loss_pixel = loss_fn(canvas, images)
            loss_perceptual_val = perceptual_loss(canvas, images)
            loss = loss_pixel + 0.5 * loss_perceptual_val  # stronger perceptual weight

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

        # Display first canvas vs target every 5 epochs
        if epoch % 5 == 0:
            target_img = images[0].detach().cpu().permute(1,2,0).numpy()
            canvas_img = canvas[0].detach().cpu().permute(1,2,0).numpy()

            fig, axs = plt.subplots(1,2,figsize=(8,4))
            axs[0].imshow(target_img)
            axs[0].set_title("Target")
            axs[0].axis('off')
            axs[1].imshow(canvas_img)
            axs[1].set_title("Canvas")
            axs[1].axis('off')
            plt.show()



train_painter_multi_stroke(model, train_loader, num_epochs=50, optimizer=optimizer,
              loss_fn=loss_fn, perceptual_loss=perceptual_loss, num_strokes=50, device=device)

# input_tensor = torch.input_tensor.to(device)

# print("Generating model...")

# network = ConvolutionalNet()

# model = network.to(device)

# print("Model generated:")
# print(model)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# for epoch in range(2):  # loop over the dataset multiple times

#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#             running_loss = 0.0

# print('Finished Training')