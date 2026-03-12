import torch
import torch.nn as nn


# Maps input image x -> latent parameters (μ, logσ²)
class Encoder_lin(nn.Module):
    def __init__(self, input_size=784, hidden_size=600, latent_size=40):
        super(Encoder_lin, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.mean_layer = nn.Linear(hidden_size, latent_size)
        self.logvar_layer = nn.Linear(hidden_size, latent_size)
        self.activation = nn.ReLU()

    def flatten(self, x):
        N = x.shape[0]  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

    def forward(self, x):
        h = self.activation(self.input_layer(x))
        mu = self.mean_layer(h)
        log_var = self.logvar_layer(h)
        return mu, log_var

class Decoder_lin(nn.Module):
    def __init__(self, latent_size=40, hidden_size=600, output_size=784):
        super(Decoder_lin, self).__init__()

        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

    def forward(self, z):
        h = self.activation(self.latent_to_hidden(z))
        reconstruction = self.output_activation(self.hidden_to_output(h))
        return reconstruction

class Encoder_CVAE_lin(nn.Module):
    def __init__(self, input_size=794, hidden_size=600, latent_size=40, num_classes=10):
        super(Encoder_CVAE_lin, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.mean_layer = nn.Linear(hidden_size, latent_size)
        self.logvar_layer = nn.Linear(hidden_size, latent_size)
        self.activation = nn.ReLU()

    def flatten(self, x):
        N = x.shape[0]  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

    def forward(self, x,y):
        x = torch.cat([x,y],dim=1)
        h = self.activation(self.input_layer(x))
        mu = self.mean_layer(h)
        log_var = self.logvar_layer(h)
        return mu, log_var

class Decoder_CVAE_lin(nn.Module):
    def __init__(self, latent_size=50, hidden_size=600, output_size=784):
        super(Decoder_CVAE_lin, self).__init__()

        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

    def forward(self, z,y):
        z = torch.cat([z,y],dim=1)
        h = self.activation(self.latent_to_hidden(z))
        reconstruction = self.output_activation(self.hidden_to_output(h))
        return reconstruction

class Encoder_conv(nn.Module):
    def __init__(self, input_size=784, hidden_size=600, latent_size=40):
        super(Encoder_conv, self).__init__()

        #self.input_layer = nn.Linear(input_size, hidden_size)
        #self.mean_layer = nn.Linear(hidden_size, latent_size)
        #self.logvar_layer = nn.Linear(hidden_size, latent_size)
        self.mean_layer = nn.Linear(128, latent_size)
        self.logvar_layer = nn.Linear(128, latent_size)

        self.activation = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 32, 4,2,1)
        self.conv2 = nn.Conv2d(32, 64, 4,2,1)
        self.conv3 = nn.Conv2d(64, 128, 7)

    def flatten(self, x):
        N = x.shape[0]  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

    def forward(self, x):
        #h = self.activation(self.input_layer(x))
        #mu = self.mean_layer(h)
        #log_var = self.logvar_layer(h)
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(x1))
        x3 = self.activation(self.conv3(x2))
        x4 = self.flatten(x3)
        mu = self.mean_layer(x4)
        log_var = self.logvar_layer(x4)

        return mu, log_var

#Decode network
class Decoder_conv(nn.Module):
    def __init__(self, latent_size=40, hidden_size=600, output_size=784):
        super(Decoder_conv, self).__init__()

        #self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        #self.hidden_to_output = nn.Linear(hidden_size, output_size)
        self.fc1 = nn.Linear(latent_size, 128)
        self.conv1 = nn.ConvTranspose2d(128, 64, 7)
        self.conv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.conv3 = nn.ConvTranspose2d(32, 1, 4, 2, 1)
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

    def forward(self, z):
        #h = self.activation(self.latent_to_hidden(z))
        #reconstruction = self.output_activation(self.hidden_to_output(h))
        x1 = self.fc1(z)
        x2 = x1.view(-1, 128, 1, 1)
        x3 = self.activation(self.conv1(x2))
        x4 = self.activation(self.conv2(x3))
        reconstruction = self.output_activation(self.conv3(x4))
        return reconstruction


class VAE_conv(nn.Module):
    def __init__(self):
        super(VAE_conv, self).__init__()

        self.encoder = Encoder_conv()
        self.decoder = Decoder_conv()


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

class VAE_linear(nn.Module):
    def __init__(self):
        super(VAE_linear, self).__init__()

        self.encoder = Encoder_lin()
        self.decoder = Decoder_lin()


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


class CVAE_linear(nn.Module):
    def __init__(self):
        super(CVAE_linear, self).__init__()

        self.encoder = Encoder_CVAE_lin()
        self.decoder = Decoder_CVAE_lin()


    #Reparameterization trick
    def sample_z(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        return z

    #Forward pass
    def forward(self, x,y):
        mu, log_var = self.encoder(x,y)
        z = self.sample_z(mu, log_var)
        x_reconstructed = self.decoder(z,y)
        return x_reconstructed, mu, log_var


