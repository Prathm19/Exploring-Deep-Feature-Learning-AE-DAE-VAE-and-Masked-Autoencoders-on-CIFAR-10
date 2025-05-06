# ----------------------
# Notebook Cell 1: Imports & Setup
# ----------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------
# Notebook Cell 2: Data Loading with Noise Injection
# ----------------------
def add_gaussian_noise(img, mean=0., std=0.1):
    noise = torch.randn_like(img) * std + mean
    noisy_img = img + noise
    return torch.clamp(noisy_img, 0., 1.)

transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# ----------------------
# Notebook Cell 3: Denoising Autoencoder Architecture
# ----------------------
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 16x16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 8x8
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 4x4
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=1, padding=1),  # 8x8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1, padding=1),   # 16x16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, output_padding=1, padding=1),    # 32x32
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out, latent

model = DenoisingAutoencoder().to(device)

# ----------------------
# Notebook Cell 4: Training Function
# ----------------------
def train_dae(model, noise_std, num_epochs=20):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_errors = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for inputs, _ in trainloader:
            inputs = inputs.to(device)
            noisy_inputs = add_gaussian_noise(inputs, std=noise_std).to(device)

            outputs, _ = model(noisy_inputs)
            loss = criterion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)

        avg_loss = epoch_loss / len(trainloader.dataset)
        train_errors.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return train_errors

# ----------------------
# Notebook Cell 5: Train DAE with Different Noise Levels
# ----------------------
noise_levels = [0.1, 0.3, 0.5]
all_errors = {}

for noise_std in noise_levels:
    print(f"\nTraining DAE with Gaussian noise std = {noise_std}")
    model = DenoisingAutoencoder().to(device)
    errors = train_dae(model, noise_std=noise_std, num_epochs=20)
    all_errors[noise_std] = errors

# ----------------------
# Notebook Cell 6: Plot Errors
# ----------------------
plt.figure(figsize=(10,6), dpi=300)
for noise_std, errors in all_errors.items():
    plt.plot(errors, label=f"Noise std = {noise_std}")

plt.xlabel("Epochs")
plt.ylabel("Average MSE")
plt.title("Training Error vs Epochs for Denoising Autoencoder")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("dae_training_error.png", dpi=300)
plt.show()

# ----------------------
# Notebook Cell 7: Visualize Clean, Noisy, and Reconstructed Images
# ----------------------
def imshow_tensor(img):
    npimg = img.numpy().transpose(1, 2, 0)
    return np.clip(npimg, 0, 1)

model.eval()
dataiter = iter(testloader)
images, _ = next(dataiter)
noisy_images = add_gaussian_noise(images, std=0.3)

with torch.no_grad():
    reconstructed, _ = model(noisy_images.to(device))

fig, axes = plt.subplots(5, 3, figsize=(10, 8), dpi=300)
for i in range(5):
    axes[i, 0].imshow(imshow_tensor(images[i]))
    axes[i, 0].set_title("Original")
    axes[i, 0].axis('off')

    axes[i, 1].imshow(imshow_tensor(noisy_images[i]))
    axes[i, 1].set_title("Noisy")
    axes[i, 1].axis('off')

    axes[i, 2].imshow(imshow_tensor(reconstructed.cpu()[i]))
    axes[i, 2].set_title("Reconstructed")
    axes[i, 2].axis('off')

plt.tight_layout()
plt.savefig("dae_visualization.png", dpi=300)
plt.show()
