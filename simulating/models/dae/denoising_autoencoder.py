import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam

class DenoisingAutoencoder(pl.LightningModule):
    def __init__(self, original_dim, hidden_dims, encoding_dim, dropout_rate=0.2, lr=1e-3, activation=nn.ReLU, output_activation=nn.Sigmoid, noise_std=0.1):
        """
        Args:
            original_dim (int): Number of input features.
            hidden_dims (list[int]): List of hidden layer sizes (encoder).
            encoding_dim (int): Size of latent space.
            dropout_rate (float): Dropout applied between layers.
            lr (float): Learning rate.
            activation (torch.nn.Module): Activation function.
            output_activation (torch.nn.Module): Final decoder activation.
            noise_std (float): Standard deviation of Gaussian noise during training.
        """
        super().__init__()
        self.save_hyperparameters() # Saves all init arguments to self.hparams

        # ---- Encoder ----
        encoder_layers = []
        in_dim = original_dim
        for out_dim in hidden_dims:
            encoder_layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                activation(),
                nn.Dropout(dropout_rate),
            ]
            in_dim = out_dim
        encoder_layers.append(nn.Linear(in_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # ---- Decoder ----
        decoder_layers = []
        in_dim = encoding_dim
        for out_dim in reversed(hidden_dims):
            decoder_layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                activation(),
                # nn.Dropout(dropout_rate),
            ]
            in_dim = out_dim
        decoder_layers.append(nn.Linear(in_dim, original_dim))
        decoder_layers.append(output_activation())
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Mean Squared Error for continuous feature
        self.criterion = nn.MSELoss()

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed

    def training_step(self, batch, batch_idx):
        x_clean, = batch

        # Add noise
        noise = torch.randn_like(x_clean) * self.hparams.noise_std
        x_corrupted = x_clean + noise

        reconstructed = self(x_corrupted)
        loss = self.criterion(reconstructed, x_clean)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, = batch
        reconstructed = self(x)
        loss = self.criterion(reconstructed, x)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)

    def encode(self, x):
        """Return latent representation of input"""
        self.eval()
        with torch.no_grad():
            return self.encoder(x)

    def decode(self, z):
        """Return reconstruction from latent space"""
        self.eval()
        with torch.no_grad():
            return self.decoder(z)
