# Variational Autoencoder Reimplementation (Fashion-MNIST)

This project contains a **from-scratch implementation of a Variational Autoencoder (VAE)** using PyTorch.
The model learns a latent representation of the Fashion-MNIST dataset and can generate new clothing images by sampling from the learned latent space.

### File Descriptions

**model.py**

Defines the Variational Autoencoder architecture.
Includes:

* Encoder network
* Latent mean and variance layers
* Reparameterization sampling step
* Decoder network
* Forward pass of the model

---

**dataset.py**

Handles loading the Fashion-MNIST dataset and creating a PyTorch `DataLoader` used during training.

---

**train.py**

Runs the training process for the VAE.
Responsibilities include:

* Initializing the model
* Computing the ELBO loss (reconstruction + KL divergence)
* Performing gradient updates using Adam optimizer
* Saving the trained model weights

---

**generate.py**

Loads the trained model and generates new images by sampling from the latent space.
The generated images are displayed using matplotlib.

---

# Installation

Install dependencies:

```
pip install -r requirements.txt
```

---

# Training the Model

Run the training script:

```
python train.py
```
