import torch
import matplotlib.pyplot as plt
from model import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE().to(device)
model.load_state_dict(torch.load("vae_fashion.pth"))
model.eval()

with torch.no_grad():

    z = torch.randn(16, 20).to(device)

    generated = model.decoder(z)
    generated = generated.view(-1, 28, 28)

fig, axes = plt.subplots(4, 4)

for i in range(16):

    row = i // 4
    col = i % 4

    axes[row, col].imshow(generated[i].cpu())
    axes[row, col].axis("off")

plt.show()
