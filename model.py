import torch
import torch.nn as nn


# Maps input image x -> latent parameters (μ, logσ²)
class Encoder(nn.Module):
    def __init__(self, input_size=784, hidden_size=400, latent_size=20):
        super(Encoder, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.mean_layer = nn.Linear(hidden_size, latent_size)
        self.logvar_layer = nn.Linear(hidden_size, latent_size)

        self.activation = nn.ReLU()

    def forward(self, x):
        h = self.activation(self.input_layer(x))
        mu = self.mean_layer(h)
        log_var = self.logvar_layer(h)

        return mu, log_var

#Decode network
class Decoder(nn.Module):
    def __init__(self, latent_size=20, hidden_size=400, output_size=784):
        super(Decoder, self).__init__()

        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)

        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

    def forward(self, z):
        h = self.activation(self.latent_to_hidden(z))
        reconstruction = self.output_activation(self.hidden_to_output(h))
        return reconstruction


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()


    #Reparameterization trick
    def sample_z(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        return z

    #Forward pass
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sample_z(mu, log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, log_var
