import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import numpy as np

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128, shuffle=False)

# Autoencoder definition
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),  # [B, 64, 16, 16]
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # [B, 128, 8, 8]
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # [B, 256, 4, 4]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # [B, 128, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # [B, 64, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),  # [B, 3, 32, 32]
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 20
train_errors = []

for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    for images, _ in trainloader:
        images = images.to(device)
        outputs, _ = model(images)
        loss = criterion(outputs, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(trainloader)
    train_errors.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Error vs. epochs plot
plt.figure()
plt.plot(range(1, num_epochs+1), train_errors)
plt.xlabel("Epochs")
plt.ylabel("Average MSE Loss")
plt.title("Training Error vs Epochs")
plt.grid(True)
plt.savefig("mse_loss_curve.png")
plt.show()

# Evaluation: SSIM, PSNR, MAE, MSE
model.eval()
mse_total, mae_total, ssim_total, psnr_total = 0, 0, 0, 0
n_samples = 0

with torch.no_grad():
    for images, _ in testloader:
        images = images.to(device)
        outputs, _ = model(images)
        images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
        outputs_np = outputs.cpu().numpy().transpose(0, 2, 3, 1)

        for i in range(images_np.shape[0]):
            orig = (images_np[i] * 0.5 + 0.5).clip(0, 1)
            recon = (outputs_np[i] * 0.5 + 0.5).clip(0, 1)
            mse_val = np.mean((orig - recon) ** 2)
            mae_val = np.mean(np.abs(orig - recon))
            ssim_val = ssim(orig, recon, channel_axis=-1)
            psnr_val = psnr(orig, recon)

            mse_total += mse_val
            mae_total += mae_val
            ssim_total += ssim_val
            psnr_total += psnr_val
            n_samples += 1

print(f"Avg MSE: {mse_total/n_samples:.4f}")
print(f"Avg MAE: {mae_total/n_samples:.4f}")
print(f"Avg SSIM: {ssim_total/n_samples:.4f}")
print(f"Avg PSNR: {psnr_total/n_samples:.2f} dB")

# t-SNE visualization
z_all = []
y_all = []

with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        _, z = model(images)
        z_flat = z.view(z.size(0), -1)
        z_all.append(z_flat.cpu().numpy())
        y_all.append(labels.numpy())

z_all = np.concatenate(z_all, axis=0)
y_all = np.concatenate(y_all, axis=0)

z_embedded = TSNE(n_components=2).fit_transform(z_all)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(z_embedded[:, 0], z_embedded[:, 1], c=y_all, cmap='tab10', s=5)
plt.legend(*scatter.legend_elements(), title="Classes")
plt.title("t-SNE of Latent Space")
plt.savefig("tsne_latent.png")
plt.show()

# Compare clean vs noisy inputs
noisy_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.2 * torch.randn_like(x)),
    transforms.Normalize((0.5,), (0.5,))
])

test_noisy = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=noisy_transform)
test_noisy_loader = DataLoader(test_noisy, batch_size=128, shuffle=False)

z_clean, z_noisy = [], []

with torch.no_grad():
    for (clean, _), (noisy, _) in zip(testloader, test_noisy_loader):
        clean = clean.to(device)
        noisy = noisy.to(device)
        _, z_c = model(clean)
        _, z_n = model(noisy)
        z_clean.append(z_c.view(z_c.size(0), -1).cpu().numpy())
        z_noisy.append(z_n.view(z_n.size(0), -1).cpu().numpy())

z_clean = np.concatenate(z_clean, axis=0)
z_noisy = np.concatenate(z_noisy, axis=0)

pca = PCA(n_components=2)
z_clean_2d = pca.fit_transform(z_clean)
z_noisy_2d = pca.transform(z_noisy)

plt.figure(figsize=(8, 6))
plt.scatter(z_clean_2d[:, 0], z_clean_2d[:, 1], label='Clean', alpha=0.5, s=5)
plt.scatter(z_noisy_2d[:, 0], z_noisy_2d[:, 1], label='Noisy', alpha=0.5, s=5)
plt.legend()
plt.title("PCA of Latent Space: Clean vs Noisy Inputs")
plt.savefig("pca_clean_vs_noisy.png")
plt.show()

# Visualize some reconstructions
examples = iter(testloader)
images, _ = next(examples)
images = images.to(device)
recon, _ = model(images)

images = images[:8].cpu().numpy()
recon = recon[:8].detach().cpu().numpy()

plt.figure(figsize=(16, 4))
for i in range(8):
    # Original
    ax = plt.subplot(2, 8, i + 1)
    plt.imshow(np.transpose(images[i] * 0.5 + 0.5, (1, 2, 0)))
    plt.axis('off')
    # Reconstructed
    ax = plt.subplot(2, 8, i + 9)
    plt.imshow(np.transpose(recon[i] * 0.5 + 0.5, (1, 2, 0)))
    plt.axis('off')

plt.suptitle("Original vs Reconstructed Images")
plt.savefig("reconstructions.png")
plt.show()
