import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM, FrechetInceptionDistance as FID

from loss import PerceptualLoss
from network import NeuralPainter
# from renderer import DifferentiableRenderer

IMG_SIZE = 64
BATCH_SIZE = 12
DISPLAY_EVERY_N_EPOCHS = 300

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("images/coco/test_dataset", transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------------
# Stroke rendering
# -------------------------------
def render_stroke(canvas, x, y, radius, color):
    """
    Draw a slightly elongated, rotated stroke on the canvas.
    
    canvas: [B, C, H, W]
    x, y: [B] normalized center coordinates [0,1]
    radius: [B] base radius (used for both axes, elongation added internally)
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

    # random small orientation for each stroke
    angle = torch.zeros(B, device=device)
    cos_a = torch.cos(angle)[:, None, None]
    sin_a = torch.sin(angle)[:, None, None]

    # rotate coordinates
    xr = cos_a * dx + sin_a * dy
    yr = -sin_a * dx + cos_a * dy

    # make ellipse: radius for x, 1.5*radius for y (elongation)
    radius = radius.clamp(0.05, 0.25)
    rx = radius[:, None, None].clamp(0.05, 0.25)
    ry = (1.5 * radius[:, None, None]).clamp(0.05, 0.35)


    mask = torch.exp(-(xr**2 / (2*rx**2) + yr**2 / (2*ry**2)))

    # normalize and scale opacity
    mask = mask / mask.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    mask = mask * 1.1
    mask = mask.unsqueeze(1)  # [B,1,H,W]

    color = color[:, :, None, None]  # [B,C,1,1]

    new_canvas = canvas * (1 - mask) + color * mask
    return new_canvas



# -------------------------------
# Training loop (multi-stroke)
# -------------------------------
def train_painter_multi(model, train_loader, num_epochs, optimizer, loss_fn, perceptual_loss,
                        num_strokes=50, strokes_per_step=5, device='cuda', stroke_noise=False):
    model.to(device)

    # Establish SSIM and FID metrics
    ssim_metric = SSIM(data_range=1.0).to(device)
    fid_metric = FID(feature=2048).to(device)

    for epoch in range(num_epochs):
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            canvas = 0.5 * torch.ones_like(images).to(device)

            optimizer.zero_grad()

            for stroke_idx in range(num_strokes):
                # Predict multiple strokes per image
                stroke_params = model(torch.cat([images, canvas], dim=1))
                B = images.size(0)
                stroke_params = stroke_params.view(B, strokes_per_step, 6)  # [B, N, 6]

                for n in range(strokes_per_step):
                    x, y, radius, r, g, b = stroke_params[:, n, :].split(1, dim=1)

                    if stroke_noise:
                        x = x + 0.05 * torch.randn_like(x)
                        y = y + 0.05 * torch.randn_like(y)
                        radius = radius + 0.03 * torch.randn_like(radius)
                        r = r + 0.1 * torch.randn_like(r)
                        g = g + 0.1 * torch.randn_like(g)
                        b = b + 0.1 * torch.randn_like(b)

                    # Clamp parameters
                    x = x.clamp(0,1)
                    y = y.clamp(0,1)
                    radius = radius.clamp(0.01,0.2)
                    r = r.clamp(0,1)
                    g = g.clamp(0,1)
                    b = b.clamp(0,1)

                    color = torch.cat([r,g,b], dim=1)

                    # Render stroke
                    canvas = render_stroke(canvas, x.squeeze(1), y.squeeze(1), radius.squeeze(1), color)

            # Update FID metric
            real_batch = (images * 255).clamp(0,255).byte()
            fake_batch = (canvas * 255).clamp(0,255).byte()
            
            fid_metric.update(real_batch, real=True)
            fid_metric.update(fake_batch, real=False)

            # There is a chance that the following is the correct formatting if this program doesn't handle Byte Tensors
            # fid_metric.update(images.float(), real=True)
            # fid_metric.update(canvas.float(), real=False)

            # Compute loss
            loss_pixel = loss_fn(canvas, images)
            loss_perceptual_val = perceptual_loss(canvas, images)
            loss = loss_pixel + 0.5 * loss_perceptual_val

            loss.backward()
            optimizer.step()

        # Compute SSIM of the last batch and FID for the entire epoch
        ssim_val = SSIM(data_range=1.0).to(device)(canvas, images).item()
        fid_val = fid_metric.compute().item()
        
        # Print epoch metrics
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        print(f"   SSIM: {ssim_val:.4f} | FID: {fid_val:.2f}")

        # Reset FID for the next epoch
        fid_metric.reset()

        # Display progress after a certain number of epochs
        if epoch % DISPLAY_EVERY_N_EPOCHS == 0:
            target_img = images[0].detach().cpu().permute(1,2,0).numpy()
            canvas_img = canvas[0].detach().cpu().permute(1,2,0).numpy()

            _, axs = plt.subplots(1,2,figsize=(8,4))
            axs[0].imshow(target_img)
            axs[0].set_title("Target")
            axs[0].axis('off')
            axs[1].imshow(canvas_img)
            axs[1].set_title("Canvas")
            axs[1].axis('off')
            plt.show()

# -------------------------------
# Train
# -------------------------------
if __name__ == "__main__":
    strokes_per_step = 5
    model = NeuralPainter(strokes_per_step=strokes_per_step).to(device)
    loss_fn = nn.MSELoss()
    perceptual_loss = PerceptualLoss()  # replace with your real perceptual loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 301
    num_strokes = 50

    print("Beginning training...")
    train_painter_multi(model, train_loader, num_epochs, optimizer, loss_fn,
                        perceptual_loss, num_strokes=num_strokes,
                        strokes_per_step=strokes_per_step, device=device)