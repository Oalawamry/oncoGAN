import sys
sys.path.append('/oncoGAN/models')

import click
import pickle
import pandas as pd
import torch
from dae.denoising_autoencoder import DenoisingAutoencoder


@click.command()
@click.option('--model', 'click_model',
              type=click.Choice(['driver_profile', 'genomic_profile']),
              required = True,
              help='Model to use for reconstruction')
def dae_reconstruction(click_model):

    """
    Script to reconstruct diffusion latent spaces using the DAE model
    """

    # Read input data from parent script
    z_serialized = sys.stdin.buffer.read()
    z = pickle.loads(z_serialized)

    # Load the model
    if click_model == 'driver_profile':
        checkpoint = torch.load("/oncoGAN/trained_models/driver_profile/driver_profile_autoencoder.pth", map_location="cpu", weights_only=False)
    elif click_model == 'genomic_profile':
        checkpoint = torch.load("/oncoGAN/trained_models/positional_pattern/genomic_position_autoencoder.pth", map_location="cpu", weights_only=False)
    model = DenoisingAutoencoder(**checkpoint["hparams"])
    model.load_state_dict(checkpoint["state_dict"])
    scaler = checkpoint["scaler"]

    # Reconstruct the original input
    z_tensor = torch.tensor(z.values, dtype=torch.float32)
    reconstructed = model.decode(z_tensor).detach().numpy()
    reconstructed = scaler.inverse_transform(reconstructed)
    stdout_dict = {'df': reconstructed.tolist(), 'columns': list(scaler.feature_names_in_)}

    # Return the data through stdout
    sys.stdout.buffer.write(pickle.dumps(stdout_dict))

if __name__ == '__main__':
    dae_reconstruction()
