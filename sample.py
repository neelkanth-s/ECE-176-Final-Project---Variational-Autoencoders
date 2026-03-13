import torch
import matplotlib.pyplot as plt
from model import VAE_linear
from model import VAE_conv
from model import CVAE_linear

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
use = input("Enter model 1,2 or 3 to use: ")
if use == "1":
    model = CVAE_linear().to(device)
    model.load_state_dict(torch.load("vae_fashion.pth"))
    model.eval()

    # Generate new samples
    with torch.no_grad():

        # Sample latent variables (sampling 25 vectors from the dataset)
        z = torch.randn(36, 40).to(device)
        y = torch.zeros(36, 10).to(device)
        y[:, 8] = 1
        # Decode latent variables into images
        generated = model.decoder(z,y)
        generated = generated.view(-1, 28, 28)

    fig, axes = plt.subplots(6, 6)

    for i in range(36):

        row = i // 6
        col = i % 6

        axes[row, col].imshow(generated[i].cpu(), cmap="gray")
        axes[row, col].axis("off")

    plt.show()
if use == "2":
    model = VAE_linear().to(device)
    model.load_state_dict(torch.load("vae_fashion.pth"))
    model.eval()

    # Generate new samples
    with torch.no_grad():

        # Sample latent variables (sampling 25 vectors from the dataset)
        z = torch.randn(36, 40).to(device)
        # Decode latent variables into images
        generated = model.decoder(z)
        generated = generated.view(-1, 28, 28)

    fig, axes = plt.subplots(6, 6)

    for i in range(36):
        row = i // 6
        col = i % 6

        axes[row, col].imshow(generated[i].cpu(), cmap="gray")
        axes[row, col].axis("off")

    plt.show()
if use == "3":
    model = VAE_conv().to(device)
    model.load_state_dict(torch.load("vae_fashion.pth"))
    model.eval()

    # Generate new samples
    with torch.no_grad():

        # Sample latent variables (sampling 25 vectors from the dataset)
        z = torch.randn(36, 40).to(device)
        # Decode latent variables into images
        generated = model.decoder(z)
        generated = generated.view(-1, 28, 28)

    fig, axes = plt.subplots(6, 6)

    for i in range(36):
        row = i // 6
        col = i % 6

        axes[row, col].imshow(generated[i].cpu(), cmap="gray")
        axes[row, col].axis("off")

    plt.show()