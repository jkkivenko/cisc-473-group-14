import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.fid import FrechetInceptionDistance as FID

from loss import EarlyStopping, PerceptualLoss
from network import NeuralPainter
from renderer import Renderer

IMG_SIZE = 64
BATCH_SIZE = 12
DISPLAY_EVERY_N_EPOCHS = 1
PERCEPTUAL_LOSS_AMOUNT = 0.2

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("images/coco/test_dataset", transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(torch.cuda.is_available())
early_stopping = EarlyStopping(tolerance=5, min_delta=10)

def train_one_epoch(model, train_loader, loss_fn, perceptual_loss, optimizer, num_strokes):
    train_loss = 0

    # Establish SSIM and FID metrics
    fid_metric = FID(feature=2048).to(device)
    model.train()
    # Initialize renderer
    rend = Renderer(device)
    for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            rend.initialize_canvas(images)
            optimizer.zero_grad()

            # Generate num_strokes strokes for each image
            for stroke_idx in range(num_strokes):
                stroke_params = model(torch.cat([images, rend.canvas], dim=1))
                rend.render_stroke(stroke_params)

            # Update FID metric
            real_batch = (images * 255).clamp(0,255).byte()
            fake_batch = (rend.canvas * 255).clamp(0,255).byte()
            
            fid_metric.update(real_batch, real=True)
            fid_metric.update(fake_batch, real=False)

            # There is a chance that the following is the correct formatting if this program doesn't handle Byte Tensors
            # fid_metric.update(images.float(), real=True)
            # fid_metric.update(canvas.float(), real=False)

            # Compute loss
            loss_pixel = loss_fn(rend.canvas, images)
            loss_perceptual_val = perceptual_loss(rend.canvas, images)
            loss = (1-PERCEPTUAL_LOSS_AMOUNT) * 1000 * loss_pixel + PERCEPTUAL_LOSS_AMOUNT * loss_perceptual_val

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

    # Compute SSIM of the last batch and FID for the entire epoch
    ssim_val = SSIM(data_range=1.0).to(device)(rend.canvas, images).item()
    fid_val = fid_metric.compute().item()
    # Print SSIM / FID metrics for epoch
    print(f"   SSIM: {ssim_val:.4f} | FID: {fid_val:.2f}")

    # Reset FID for the next epoch
    fid_metric.reset()
    
    return train_loss / len(train_loader)

def validate_one_epoch(model, val_loader, loss_fn, perceptual_loss, num_strokes):
    val_loss = 0
    model.eval()
    # Initialize renderer
    rend = Renderer(device)
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(val_loader):
                print("Val: " + str(batch_idx))
                images = images.to(device)
                rend.initialize_canvas(images)
                for stroke_idx in range(num_strokes):
                    # Predict multiple strokes per image
                    stroke_params = model(torch.cat([images, rend.canvas], dim=1))
                    # print(f"{stroke_params=}")
                    rend.render_stroke(stroke_params)

                # Compute loss
                loss_pixel = loss_fn(rend.canvas, images)
                loss_perceptual_val = perceptual_loss(rend.canvas, images)
                val_loss = (1-PERCEPTUAL_LOSS_AMOUNT) * 1000 * loss_pixel + PERCEPTUAL_LOSS_AMOUNT * loss_perceptual_val
    return val_loss / len(val_loader)

# -------------------------------
# Training loop (multi-stroke)
# -------------------------------
def train_painter_multi(model, train_loader, val_loader, num_epochs, optimizer, loss_fn, perceptual_loss,
                        num_strokes=50, stroke_noise=False):
    model.to(device)

    train_losses = []
    val_losses = []

    loss_fig = plt.figure()
    loss_axs = loss_fig.add_axes((0,0,1,1))
    loss_axs.set_xlim(right=num_epochs)
    loss_axs.plot(0)
    _, canvas_axs = plt.subplots(1,3,figsize=(10,4))
    canvas_axs[0].set_title("Target")
    canvas_axs[0].axis('off')
    canvas_axs[1].set_title("Model View")
    canvas_axs[1].axis('off')

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, perceptual_loss, optimizer, num_strokes)
        val_loss = validate_one_epoch(model, val_loader, loss_fn, perceptual_loss, num_strokes)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        loss_axs.plot(train_losses, label='Training Loss', color='blue', linestyle='-')
        loss_axs.plot(val_losses, label='Validation Loss', color='red', linestyle='--')
        loss_axs.plot()

        # early_stopping(train_loss, val_loss)
        # if early_stopping.early_stop:
        #      print("Stopping early to prevent overfitting")
        #      break

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")

        # Display progress after a certain number of epochs
        if epoch % DISPLAY_EVERY_N_EPOCHS == 0:
            shown_image = random.choice(train_loader.dataset)[0]
            shown_image = torch.unsqueeze(shown_image, 0) # simulates a batch size of 1

            canvas_axs[2].clear()
            canvas_axs[2].set_title("Rendered Output")
            canvas_axs[2].axis('off')
            canvas_axs[2].set_xlim(-32, 32)
            canvas_axs[2].set_ylim(-32, 32)

            canvas_axs[0].imshow(shown_image[0].permute(1,2,0).numpy())

            rend = Renderer(device)
            rend.initialize_canvas(shown_image)
            rend.initialize_nondiff_canvas(canvas_axs[2], shown_image)
            # Predict multiple strokes per image
            for stroke_idx in range(num_strokes):
                stroke_params = model(torch.cat([shown_image, rend.canvas], dim=1))
                print(stroke_params)
                rend.render_stroke(stroke_params)
                rend.render_nondifferentiable(canvas_axs[2], torch.squeeze(stroke_params))
            canv = rend.canvas[0].detach().cpu().permute(1,2,0).numpy()
            canvas_axs[1].imshow(canv)
        plt.show(block=(epoch>290))
        plt.pause(0.5)


def sanity_check_loss(stroke_params):
    image = dataset[5][0]
    loss_fn = nn.MSELoss().to(device)
    perceptual_loss = PerceptualLoss().to(device)

    _, canvas_axs = plt.subplots(1,3,figsize=(10,4))
    canvas_axs[0].set_title("Target")
    canvas_axs[0].axis('off')
    canvas_axs[0].imshow(image.detach().cpu().permute(1,2,0).numpy())
    canvas_axs[1].set_title("Model Output")
    canvas_axs[1].axis('off')
    canvas_axs[2].set_title("Rendered Output")
    canvas_axs[2].axis('off')
    canvas_axs[2].set_xlim(-32, 32)
    canvas_axs[2].set_ylim(-32, 32)

    rend = Renderer(device)
    image = torch.unsqueeze(image, 0)
    rend.initialize_canvas(image)
    rend.initialize_nondiff_canvas(canvas_axs[2], image)

    rend.render_stroke(stroke_params)
    rend.render_nondifferentiable(canvas_axs[2], torch.squeeze(stroke_params))
    canv = rend.canvas[0].detach().cpu().permute(1,2,0).numpy()
    canvas_axs[1].imshow(canv)

    loss_pixel = loss_fn(rend.canvas, image)
    loss_perceptual_val = perceptual_loss(rend.canvas, image)
    loss = (1-PERCEPTUAL_LOSS_AMOUNT) * 1000 * loss_pixel + PERCEPTUAL_LOSS_AMOUNT * loss_perceptual_val

    print(f"Loss is {loss} with stroke parameters {stroke_params}")
    plt.show()

# -------------------------------
# Train
# -------------------------------
if __name__ == "__main__":
    model = NeuralPainter().to(device)
    loss_fn = nn.MSELoss().to(device)
    perceptual_loss = PerceptualLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    num_epochs = 301
    num_strokes = 3

    # # This proves that the loss/renderer is giving reasonable numbers (when the test image is a mostly white scene with a skiier)
    # Most of these were made before the x1y1 change so they don't work anymore
    # # Small red line
    # sanity_check_loss(torch.tensor([[0.5, 0.5, 0.25, 0, 1, 0, 0, 1]])) # 1522
    # # Small orange(?) line
    # sanity_check_loss(torch.tensor([[0.5, 0.5, 0.25, 0, 0.682, 0.545, 0.227, 1]])) # 881
    # # Big orange(?) line
    # sanity_check_loss(torch.tensor([[0.5, 0.5, 1, 0, 0.682, 0.545, 0.227, 1]])) # 810
    # # Big light orange(?) line
    # sanity_check_loss(torch.tensor([[0.5, 0.5, 1, 0, 0.667, 0.596, 0.376, 1]])) # 739
    # # All grey
    # sanity_check_loss(torch.tensor([[0.5, 0.5, 1, 0, 0.5, 0.5, 0.5, 1]])) # 715
    # # All white
    # sanity_check_loss(torch.tensor([[0.5, 0.5, 1, 0, 1, 1, 1, 1]])) # 665
    # # All white but with opacity 0
    # sanity_check_loss(torch.tensor([[0.5, 0.5, 1, 0, 1, 1, 1, 0]])) # 665
    # # All white but with scale 0
    # sanity_check_loss(torch.tensor([[0.5, 0.5, 0.01, 0, 1, 1, 1, 1]])) # 665
    # # All black
    # sanity_check_loss(torch.tensor([[0.5, 0.5, 1, 0, 0, 0, 0, 1]])) # 1155
    # sanity_check_loss(torch.tensor([[0.9710, 0.4963, 0.9875, 0.7357, 0.7595, 0.6755, 0.7317, 0.1268]])) # 1155
    # sanity_check_loss(torch.tensor([[0.9710, 0.4963, 0.1, 0.7357, 0.7595, 0.6755, 0.7317, 0.1268]])) # 1155

    print("Beginning training...")
    train_painter_multi(model, train_loader, val_loader, num_epochs, optimizer, loss_fn,
                        perceptual_loss, num_strokes=num_strokes)