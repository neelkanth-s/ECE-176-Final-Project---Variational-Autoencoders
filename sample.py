import torch
import matplotlib.pyplot as plt
from model import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = VAE().to(device)
model.load_state_dict(torch.load("vae_fashion.pth"))
model.eval()

# Generate new samples
with torch.no_grad():

    # Sample latent variables (sampling 16 vectors from the dataset)
    z = torch.randn(16, 20).to(device)

    # Decode latent variables into images
    generated = model.decoder(z)
    generated = generated.view(-1, 28, 28)

fig, axes = plt.subplots(4, 4)

for i in range(16):

    row = i // 4
    col = i % 4

    axes[row, col].imshow(generated[i].cpu(), cmap="gray")
    axes[row, col].axis("off")

plt.show()
