import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from loss import EarlyStopping, PerceptualLoss
from network import NeuralPainter
# from renderer import DifferentiableRenderer

IMG_SIZE = 64
BATCH_SIZE = 12
DISPLAY_EVERY_N_EPOCHS = 300

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("E:/coco copy/", transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())
early_stopping = EarlyStopping(tolerance=5, min_delta=10)

# -------------------------------
# Stroke rendering
# -------------------------------

def render_strokes_batch(canvas, stroke_params):
    """
    Vectorized multi-stroke rendering.

    canvas: [B, C, H, W]
    stroke_params: [B, N, 6] = (x, y, radius, r, g, b)
    """
    B, C, H, W = canvas.shape
    N = stroke_params.size(1)
    device = canvas.device

    # Precompute meshgrid
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, H, device=device),
        torch.linspace(0, 1, W, device=device),
        indexing='ij'
    )
    xx = xx.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    yy = yy.unsqueeze(0).unsqueeze(0)

    # Stroke params
    x = stroke_params[..., 0].unsqueeze(-1).unsqueeze(-1)
    y = stroke_params[..., 1].unsqueeze(-1).unsqueeze(-1)
    radius = stroke_params[..., 2].unsqueeze(-1).unsqueeze(-1)
    color = stroke_params[..., 3:]  # [B,N,3]

    # Clamp radius and elongate y
    radius = radius.clamp(0.05, 0.25)
    rx = radius
    ry = (radius * 1.5).clamp(0.05, 0.35)

    dx = xx - x
    dy = yy - y

    # Rotation (currently zero)
    cos_a = torch.ones_like(dx)
    sin_a = torch.zeros_like(dx)
    xr = cos_a * dx + sin_a * dy
    yr = -sin_a * dx + cos_a * dy

    # Gaussian mask
    mask = torch.exp(-(xr**2 / (2*rx**2) + yr**2 / (2*ry**2)))
    mask = mask / mask.amax(dim=(-2, -1), keepdim=True)
    mask = mask * 1.1

    # Apply strokes sequentially
    for n in range(N):
        m = mask[:, n:n+1, :, :]
        c = color[:, n, :].view(B, C, 1, 1)
        canvas = canvas * (1 - m) + c * m

    return canvas

def train_one_epoch(model, train_loader, loss_fn, perceptual_loss, optimizer, num_strokes, strokes_per_step):
    train_loss = 0
    model.train()
    for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            canvas = 0.5 * torch.ones_like(images).to(device)

            optimizer.zero_grad()

            all_strokes = []
            for stroke_idx in range(num_strokes):
                # Predict multiple strokes per image
                stroke_params = model(torch.cat([images, canvas], dim=1))
                B = images.size(0)
                stroke_params = stroke_params.view(B, strokes_per_step, 6)  # [B, N, 6]
                all_strokes.append(stroke_params)

            all_strokes = torch.cat(all_strokes, dim=1)
            # Render stroke
            canvas = render_strokes_batch(canvas, all_strokes)

            # Compute loss
            loss_pixel = loss_fn(canvas, images)
            loss_perceptual_val = perceptual_loss(canvas, images)
            loss = loss_pixel + 0.5 * loss_perceptual_val

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    return train_loss / len(train_loader)

def validate_one_epoch(model, val_loader, loss_fn, perceptual_loss, num_strokes, strokes_per_step):
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(val_loader):
                print("Val: " + str(batch_idx))
                images = images.to(device)
                canvas = 0.5 * torch.ones_like(images).to(device)
                all_strokes = []
                for stroke_idx in range(num_strokes):
                    # Predict multiple strokes per image
                    stroke_params = model(torch.cat([images, canvas], dim=1))
                    B = images.size(0)
                    stroke_params = stroke_params.view(B, strokes_per_step, 6)  # [B, N, 6]
                    all_strokes.append(stroke_params)
                    
                all_strokes = torch.cat(all_strokes, dim=1)
                canvas = render_strokes_batch(canvas, all_strokes)

                # Compute loss
                loss_pixel = loss_fn(canvas, images)
                loss_perceptual_val = perceptual_loss(canvas, images)
                val_loss += loss_pixel + 0.5 * loss_perceptual_val
    return val_loss / len(val_loader)

# -------------------------------
# Training loop (multi-stroke)
# -------------------------------
def train_painter_multi(model, train_loader, val_loader, num_epochs, optimizer, loss_fn, perceptual_loss,
                        num_strokes=50, strokes_per_step=5, stroke_noise=False):
    model.to(device)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, perceptual_loss, optimizer, num_strokes, strokes_per_step)
        val_loss = validate_one_epoch(model, val_loader, loss_fn, perceptual_loss, num_strokes, strokes_per_step)

        early_stopping(train_loss, val_loss)
        if early_stopping.early_stop:
             print("Stopping early to prevent overfitting")
             break

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")

        # Display progress after a certain number of epochs
        # if epoch % DISPLAY_EVERY_N_EPOCHS == 0:
        #     target_img = images[0].detach().cpu().permute(1,2,0).numpy()
        #     canvas_img = canvas[0].detach().cpu().permute(1,2,0).numpy()

        #     _, axs = plt.subplots(1,2,figsize=(8,4))
        #     axs[0].imshow(target_img)
        #     axs[0].set_title("Target")
        #     axs[0].axis('off')
        #     axs[1].imshow(canvas_img)
        #     axs[1].set_title("Canvas")
        #     axs[1].axis('off')
        #     plt.show()

# -------------------------------
# Train
# -------------------------------
if __name__ == "__main__":
    strokes_per_step = 5
    model = NeuralPainter(strokes_per_step=strokes_per_step).to(device)
    loss_fn = nn.MSELoss().to(device)
    perceptual_loss = PerceptualLoss().to(device)  # replace with your real perceptual loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 301
    num_strokes = 50

    print("Beginning training...")
    train_painter_multi(model, train_loader, val_loader, num_epochs, optimizer, loss_fn,
                        perceptual_loss, num_strokes=num_strokes,
                        strokes_per_step=strokes_per_step)