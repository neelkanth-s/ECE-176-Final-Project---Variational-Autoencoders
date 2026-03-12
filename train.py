import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import VAE_linear
from model import VAE_conv
from model import CVAE_linear

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
batch_size = 128
num_epochs = 20
learning_rate = 2e-3

transform = transforms.ToTensor()

# Dataset loading (Using FashionMNIST as our dataset)
train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model initialization, Using Adam to optimze parameters
model1 = CVAE_linear().to(device)
optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)

model2 = VAE_linear().to(device)
optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)

model3 = VAE_conv().to(device)
optimizer3 = optim.Adam(model3.parameters(), lr=1e-3)


#VAE loss function
def vae_loss(recon_x, x, mu, logvar):

    #Reconstruction Loss (From section 2 and 3)
    recon_loss = torch.nn.functional.binary_cross_entropy(
        recon_x,
        x,
        reduction="sum"
    )

    #KL divergence 
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    #Total Loss
    return (recon_loss + kl_loss) / batch_size


#Training Loop

use = input("Enter model 1,2 or 3 to use: ")

for epoch in range(num_epochs):

    epoch_loss = 0

    for images, labels in train_loader:
        if use == "3":
            images = images.view(-1, 1,28,28).to(device)
        else:
            images = images.view(-1, 784).to(device)
        y = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)
        if use == "1":
            optimizer1.zero_grad()
            recon_images, mu, logvar = model1(images,y)
        elif use == "2":
            optimizer2.zero_grad()
            recon_images, mu, logvar = model2(images)
        elif use == "3":
            optimizer3.zero_grad()
            recon_images, mu, logvar = model3(images)
        loss = vae_loss(recon_images, images, mu, logvar)

        loss.backward()
        if use == "1":
            optimizer1.step()
        elif use == "2":
            optimizer2.step()
        elif use == "3":
            optimizer3.step()

        epoch_loss += loss.item()

    print("Epoch:", epoch + 1, "Loss:", epoch_loss)



if use == "1":
    torch.save(model1.state_dict(), "vae_fashion.pth")
elif use == "2":
    torch.save(model2.state_dict(), "vae_fashion.pth")
elif use == "3":
    torch.save(model3.state_dict(), "vae_fashion.pth")