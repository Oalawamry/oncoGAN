import sys
sys.path.append('/oncoGAN/models')

import pandas as pd
from math import ceil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import classification_report

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import CSVLogger

from dae.denoising_autoencoder import DenoisingAutoencoder

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

df = pd.read_csv('dae_training_test.csv')
X = df.drop(columns='Tumor')
y = df['Tumor'] # Unused here, since it's unsupervised

# ---------------------------------------------------------------------
# Model options
# ---------------------------------------------------------------------

cpu = 5
epochs = 2000
batch_size = 400
input_dim = X.shape[1]
hidden_dims = [1024, 512]
encoding_dim = 256
dropout_rate = 0.3 # Default: 0.2
lr = 1e-4 # Default: 1e-3
activation = nn.ELU # Default: nn.ReLU  
output_activation = nn.Tanh # Default: nn.Sigmoid
noise_std = 0.1 # Default: 0.1

prefix = f"e{epochs}_bs{batch_size}_hd{'_'.join([str(hd) for hd in hidden_dims])}_{encoding_dim}_dr{dropout_rate}_lr{lr}_activation_{str(activation).split('.')[-1].rstrip('\'>')}_output_{str(output_activation).split('.')[-1].rstrip('\'>')}"

# ---------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

train_ds = TensorDataset(X_train_tensor)
val_ds = TensorDataset(X_val_tensor)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=cpu, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=cpu, persistent_workers=True)

steps = ceil(X_train.shape[0]/batch_size)

# ---------------------------------------------------------------------
# Trainer Setup
# ---------------------------------------------------------------------

lit_model = DenoisingAutoencoder(input_dim, hidden_dims, encoding_dim, dropout_rate, lr, activation, output_activation, noise_std)

logger = CSVLogger(".", name=prefix)

checkpoint_callback = ModelCheckpoint(
    dirpath=f"{logger.log_dir}/checkpoints",
    filename="dae-best-{epoch:02d}-{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=1
)

trainer = Trainer(
    max_epochs=epochs,
    accelerator="auto", # Device
    callbacks=[checkpoint_callback, RichProgressBar(refresh_rate=10, theme=RichProgressBarTheme(metrics_format='.5f'))],
    logger=logger,
    log_every_n_steps=steps*2
)

trainer.fit(lit_model, train_loader, val_loader)

# ---------------------------------------------------------------------
# Load the best checkpoint model
# ---------------------------------------------------------------------

device = torch.device("cpu")
best_model = DenoisingAutoencoder.load_from_checkpoint(checkpoint_callback.best_model_path)
best_model.to(device)

# ---------------------------------------------------------------------
# Latent space evaluation
# ---------------------------------------------------------------------

# Last model
train_latent = lit_model.encode(X_train_tensor)
val_latent = lit_model.encode(X_val_tensor)

lr = LogisticRegressionCV(cv=3, max_iter=100)
lr.fit(train_latent.detach().numpy(), y_train)
y_pred = lr.predict(val_latent.detach().numpy())
print_report = classification_report(y_val, y_pred, zero_division=0)
print(print_report)

# Best checkpoint model
best_train_latent = best_model.encode(X_train_tensor)
best_val_latent = best_model.encode(X_val_tensor)

lr = LogisticRegressionCV(cv=3, max_iter=100)
lr.fit(best_train_latent.detach().numpy(), y_train)
y_pred = lr.predict(best_val_latent.detach().numpy())
print_report = classification_report(y_val, y_pred, zero_division=0)
print(print_report)

# ---------------------------------------------------------------------
# Reconstructed and scale space evaluation
# ---------------------------------------------------------------------

# Last model
train_reconstructed = lit_model.decode(train_latent)
val_reconstructed = lit_model.decode(val_latent)

lr = LogisticRegression(max_iter=100)
lr.fit(train_reconstructed.detach().numpy(), y_train)
y_pred = lr.predict(val_reconstructed.detach().numpy())
print_report = classification_report(y_val, y_pred, zero_division=0)
print(print_report)

# Best checkpoint model
best_train_reconstructed = best_model.decode(best_train_latent)
best_val_reconstructed = best_model.decode(best_val_latent)

lr = LogisticRegression(max_iter=100)
lr.fit(best_train_reconstructed.detach().numpy(), y_train)
y_pred = lr.predict(best_val_reconstructed.detach().numpy())
print_report = classification_report(y_val, y_pred, zero_division=0)
print(print_report)

# ---------------------------------------------------------------------
# Reconstructed and original space evaluation
# ---------------------------------------------------------------------

# Last model
train_reconstructed_unscale = scaler.inverse_transform(train_reconstructed.detach().numpy())
val_reconstructed_unscale = scaler.inverse_transform(val_reconstructed.detach().numpy())

lr = LogisticRegression(max_iter=100)
lr.fit(train_reconstructed_unscale, y_train)
y_pred = lr.predict(val_reconstructed_unscale)
print_report = classification_report(y_val, y_pred, zero_division=0)
print(print_report)

# Best checkpoint model
best_train_reconstructed_unscale = scaler.inverse_transform(best_train_reconstructed)
best_val_reconstructed_unscale = scaler.inverse_transform(best_val_reconstructed)

lr = LogisticRegression(max_iter=100)
lr.fit(best_train_reconstructed_unscale, y_train)
y_pred = lr.predict(best_val_reconstructed_unscale)
print_report = classification_report(y_val, y_pred, zero_division=0)
print(print_report)

# ---------------------------------------------------------------------
# Save the selected model and the scaler
# ---------------------------------------------------------------------

selected_model = DenoisingAutoencoder.load_from_checkpoint(checkpoint_callback.best_model_path)
export_model = {
    "hparams": selected_model.hparams,
    "state_dict": selected_model.state_dict(),
    "scaler": scaler,
}
torch.save(export_model, "selected_model.pth")

