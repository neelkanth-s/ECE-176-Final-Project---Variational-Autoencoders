# Variational Autoencoder Reimplementation

This project is an implementation of a Variational Autoencoder (VAE).
The model is trained to learn a latent representation of image data and generate new samples by decoding points from the latent space.

The implementation follows the core ideas of Variational Autoencoders, including:

* Encoder network producing latent mean and variance
* Reparameterization trick for sampling
* Decoder network for reconstruction
* ELBO loss combining reconstruction and KL divergence

---

### model.py

Defines the neural network architectures used in the project.
This file includes implementations of:

* A **linear Conditional Variational Autoencoder (CVAE)**
* A **convolutional Variational Autoencoder**

Both models include encoder, latent sampling, and decoder components.

---

### train.py

Handles the full training process for the VAE models.
Responsibilities include:

* Loading the dataset
* Initializing the model
* Computing the VAE loss (reconstruction + KL divergence)
* Performing optimization steps
* Saving trained model weights

---

### sample.py

Generates images using a trained model.
The script samples latent vectors from a Gaussian distribution and passes them through the decoder to create new images.

The generated images are displayed in a **6×6 grid**.

---

### requirements.txt

Lists the Python dependencies required to run the project, such as:

* PyTorch
* Torchvision
* Matplotlib
* NumPy

These packages can be installed with:

```
pip install -r requirements.txt
```

---

# Running the Project

Train the model:

```
python train.py
```

Generate new samples:

```
python sample.py
```
