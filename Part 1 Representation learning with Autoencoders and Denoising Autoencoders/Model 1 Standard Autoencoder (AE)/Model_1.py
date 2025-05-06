import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------ Config ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 20
LATENT_DIM = 128

# ------------------ Data ------------------
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------ Autoencoder Model ------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # (16x16)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # (8x8)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # (4x4)
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------------------ Train ------------------
train_errors = []
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, _ in trainloader:
        inputs = inputs.to(device)
        outputs, _ = model(inputs)
        loss = criterion(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)
    train_errors.append(avg_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

# ------------------ Plot Error vs. Epochs ------------------
plt.figure(figsize=(6, 4))
plt.plot(train_errors, marker='o')
plt.title("Training Error vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("error_vs_epochs.png")
plt.show()

# ------------------ Evaluation ------------------
model.eval()
originals = []
reconstructions = []
latent_vectors = []
labels = []

with torch.no_grad():
    for inputs, targets in testloader:
        inputs = inputs.to(device)
        outputs, latents = model(inputs)
        originals.append(inputs.cpu())
        reconstructions.append(outputs.cpu())
        latent_vectors.append(latents.view(latents.size(0), -1).cpu())
        labels.append(targets)

originals = torch.cat(originals)
reconstructions = torch.cat(reconstructions)
latent_vectors = torch.cat(latent_vectors)
labels = torch.cat(labels)

# ------------------ SSIM, PSNR, MAE, MSE ------------------
ssim_vals, psnr_vals, mae_vals, mse_vals = [], [], [], []

for orig, recon in zip(originals, reconstructions):
    orig_np = orig.permute(1, 2, 0).numpy()
    recon_np = recon.permute(1, 2, 0).numpy()

    ssim_val = ssim(orig_np, recon_np, win_size=7, channel_axis=-1, data_range=1.0)
    psnr_val = psnr(orig_np, recon_np, data_range=1.0)
    mae_val = mean_absolute_error(orig_np.flatten(), recon_np.flatten())
    mse_val = mean_squared_error(orig_np.flatten(), recon_np.flatten())

    ssim_vals.append(ssim_val)
    psnr_vals.append(psnr_val)
    mae_vals.append(mae_val)
    mse_vals.append(mse_val)

print(f"\n--- Evaluation Metrics on Test Set ---")
print(f"Average SSIM: {np.mean(ssim_vals):.4f}")
print(f"Average PSNR: {np.mean(psnr_vals):.4f}")
print(f"Average MAE : {np.mean(mae_vals):.4f}")
print(f"Average MSE : {np.mean(mse_vals):.4f}")

# ------------------ Visualize Reconstructed Images ------------------
def imshow_tensor(tensor):
    npimg = tensor.numpy().transpose(1, 2, 0)
    return np.clip(npimg, 0, 1)

fig, axes = plt.subplots(5, 2, figsize=(6, 12))
for i in range(5):
    axes[i, 0].imshow(imshow_tensor(originals[i]))
    axes[i, 0].set_title("Original")
    axes[i, 0].axis('off')

    axes[i, 1].imshow(imshow_tensor(reconstructions[i]))
    axes[i, 1].set_title("Reconstructed")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.savefig("original_vs_reconstructed.png")
plt.show()

# ------------------ Latent Space Visualization ------------------
# PCA
pca = PCA(n_components=2)
latent_pca = pca.fit_transform(latent_vectors)
plt.figure(figsize=(6, 5))
plt.scatter(latent_pca[:, 0], latent_pca[:, 1], c=labels, cmap='tab10', s=10)
plt.title("Latent Space (PCA)")
plt.colorbar()
plt.tight_layout()
plt.savefig("latent_space_pca.png")
plt.show()

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
latent_tsne = tsne.fit_transform(latent_vectors)
plt.figure(figsize=(6, 5))
plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels, cmap='tab10', s=10)
plt.title("Latent Space (t-SNE)")
plt.colorbar()
plt.tight_layout()
plt.savefig("latent_space_tsne.png")
plt.show()
