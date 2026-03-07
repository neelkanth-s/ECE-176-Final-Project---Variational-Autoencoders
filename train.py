#from keras.datasets import fashion_mnist
#import tensorflow
#import matplotlib.pyplot as plt
#import numpy as np
#(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#print(x_train.shape)   # (60000, 28, 28)
#print(y_train.shape)   # (60000,)

#x_class = x_train[y_train == 0]
#fig, axes = plt.subplots(1, 5, figsize=(10,2))

#for i, ax in enumerate(axes):
#    ax.imshow(x_class[i], cmap="gray")
#    ax.axis("off")

#plt.show()

#X_train_flat = x_class.reshape(len(x_class), 28*28)
#print(np.shape(X_train_flat))
#mu = np.mean(X_train_flat,0)
#var = np.var(X_train_flat,0)
#mu = mu.reshape(28,28)
#plt.imshow(mu, cmap="gray")
#plt.show()

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
batch_size = 128
num_epochs = 20
learning_rate = 1e-3

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
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


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
    return recon_loss + kl_loss


#Training Loop
for epoch in range(num_epochs):

    epoch_loss = 0

    for images, _ in train_loader:
        images = images.view(-1, 784).to(device)

        optimizer.zero_grad()
        recon_images, mu, logvar = model(images)
        loss = vae_loss(recon_images, images, mu, logvar)

        loss.backward()
        optimizer.step()

        #forward parametrization 
        #loss.forward()

        epoch_loss += loss.item()

    print("Epoch:", epoch + 1, "Loss:", epoch_loss)

torch.save(model.state_dict(), "vae_fashion.pth")
