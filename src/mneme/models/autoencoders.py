"""Autoencoder models for field latent space analysis.

This module provides variational autoencoders (VAEs) for learning
compressed representations of bioelectric field data.
"""
from __future__ import annotations

from typing import Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass
import numpy as np

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False


@dataclass
class VAEOutput:
    """Output from VAE forward pass."""
    reconstruction: Any  # torch.Tensor
    mu: Any  # torch.Tensor
    log_var: Any  # torch.Tensor
    z: Any  # torch.Tensor


@dataclass 
class TrainingResult:
    """Result from VAE training."""
    train_losses: List[float]
    val_losses: Optional[List[float]]
    best_epoch: int
    final_loss: float


class ConvBlock(nn.Module if _TORCH_AVAILABLE else object):
    """Convolutional block with BatchNorm and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: str = "leaky_relu",
    ) -> None:
        if not _TORCH_AVAILABLE:
            return
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.Identity()
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class ConvTransposeBlock(nn.Module if _TORCH_AVAILABLE else object):
    """Transposed convolutional block for upsampling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        activation: str = "leaky_relu",
    ) -> None:
        if not _TORCH_AVAILABLE:
            return
        super().__init__()
        
        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.Identity()
    
    def forward(self, x):
        return self.activation(self.bn(self.conv_t(x)))


class FieldAutoencoder(nn.Module if _TORCH_AVAILABLE else object):
    """Convolutional Variational Autoencoder for 2D field data.
    
    This VAE learns a compressed latent representation of bioelectric
    field patterns, which can be used for:
    - Dimensionality reduction
    - Anomaly detection
    - Generative modeling
    - Feature extraction for downstream analysis
    
    Parameters
    ----------
    input_shape : Tuple[int, int]
        Shape of input field (height, width). Should be divisible by 16.
    latent_dim : int
        Dimension of latent space. Default 32.
    in_channels : int
        Number of input channels. Default 1 for single field.
    base_channels : int
        Base number of channels in first conv layer. Doubles each layer.
    architecture : str
        Architecture type: 'standard', 'deep', or 'residual'.
    beta : float
        Beta parameter for β-VAE. Higher values encourage disentanglement.
        
    Examples
    --------
    >>> import torch
    >>> from mneme.models.autoencoders import FieldAutoencoder
    >>> 
    >>> # Create VAE for 64x64 fields
    >>> vae = FieldAutoencoder(input_shape=(64, 64), latent_dim=16)
    >>> 
    >>> # Forward pass
    >>> x = torch.randn(8, 1, 64, 64)  # batch of 8 fields
    >>> output = vae(x)
    >>> 
    >>> # Access outputs
    >>> recon = output.reconstruction  # Reconstructed fields
    >>> z = output.z  # Latent representations
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        latent_dim: int = 32,
        in_channels: int = 1,
        base_channels: int = 32,
        architecture: str = "standard",
        beta: float = 1.0,
    ) -> None:
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.architecture = architecture
        self.beta = beta
        
        if not _TORCH_AVAILABLE:
            # Lightweight placeholder
            return
        
        super().__init__()
        
        h, w = input_shape
        
        # Validate input shape
        if h % 16 != 0 or w % 16 != 0:
            raise ValueError(
                f"Input shape {input_shape} must be divisible by 16. "
                f"Consider padding or resizing to {((h // 16 + 1) * 16, (w // 16 + 1) * 16)}"
            )
        
        # Calculate sizes after encoding
        self._h_encoded = h // 16
        self._w_encoded = w // 16
        self._flat_size = base_channels * 8 * self._h_encoded * self._w_encoded
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Latent space projections
        self.fc_mu = nn.Linear(self._flat_size, latent_dim)
        self.fc_log_var = nn.Linear(self._flat_size, latent_dim)
        
        # Decoder input projection
        self.fc_decode = nn.Linear(latent_dim, self._flat_size)
        
        # Build decoder
        self.decoder = self._build_decoder()
        
        # Final output layer
        self.output_conv = nn.Conv2d(base_channels, in_channels, 3, 1, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_encoder(self) -> nn.Sequential:
        """Build encoder network."""
        bc = self.base_channels
        
        layers = [
            # Input -> bc channels, stride 2 (h/2, w/2)
            ConvBlock(self.in_channels, bc, 4, 2, 1),
            # bc -> 2*bc, stride 2 (h/4, w/4)
            ConvBlock(bc, bc * 2, 4, 2, 1),
            # 2*bc -> 4*bc, stride 2 (h/8, w/8)
            ConvBlock(bc * 2, bc * 4, 4, 2, 1),
            # 4*bc -> 8*bc, stride 2 (h/16, w/16)
            ConvBlock(bc * 4, bc * 8, 4, 2, 1),
        ]
        
        if self.architecture == "deep":
            # Add extra conv layers without downsampling
            layers.extend([
                ConvBlock(bc * 8, bc * 8, 3, 1, 1),
                ConvBlock(bc * 8, bc * 8, 3, 1, 1),
            ])
        
        layers.append(nn.Flatten())
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.Sequential:
        """Build decoder network."""
        bc = self.base_channels
        
        layers = []
        
        if self.architecture == "deep":
            layers.extend([
                ConvBlock(bc * 8, bc * 8, 3, 1, 1),
                ConvBlock(bc * 8, bc * 8, 3, 1, 1),
            ])
        
        layers.extend([
            # 8*bc -> 4*bc, upsample (h/8, w/8)
            ConvTransposeBlock(bc * 8, bc * 4, 4, 2, 1),
            # 4*bc -> 2*bc, upsample (h/4, w/4)
            ConvTransposeBlock(bc * 4, bc * 2, 4, 2, 1),
            # 2*bc -> bc, upsample (h/2, w/2)
            ConvTransposeBlock(bc * 2, bc, 4, 2, 1),
            # bc -> bc, upsample (h, w)
            ConvTransposeBlock(bc, bc, 4, 2, 1),
        ])
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x: 'torch.Tensor') -> Tuple['torch.Tensor', 'torch.Tensor']:
        """Encode input to latent distribution parameters.
        
        Parameters
        ----------
        x : torch.Tensor
            Input field of shape (batch, channels, height, width).
            
        Returns
        -------
        mu : torch.Tensor
            Mean of latent distribution, shape (batch, latent_dim).
        log_var : torch.Tensor
            Log variance of latent distribution, shape (batch, latent_dim).
        """
        if not _TORCH_AVAILABLE:
            return None, None
        
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var
    
    def reparameterize(self, mu: 'torch.Tensor', log_var: 'torch.Tensor') -> 'torch.Tensor':
        """Reparameterization trick for VAE.
        
        Parameters
        ----------
        mu : torch.Tensor
            Mean of latent distribution.
        log_var : torch.Tensor
            Log variance of latent distribution.
            
        Returns
        -------
        z : torch.Tensor
            Sampled latent vector.
        """
        if not _TORCH_AVAILABLE:
            return None
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: 'torch.Tensor') -> 'torch.Tensor':
        """Decode latent vector to field reconstruction.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent vector of shape (batch, latent_dim).
            
        Returns
        -------
        reconstruction : torch.Tensor
            Reconstructed field of shape (batch, channels, height, width).
        """
        if not _TORCH_AVAILABLE:
            return None
        
        h = self.fc_decode(z)
        h = h.view(-1, self.base_channels * 8, self._h_encoded, self._w_encoded)
        h = self.decoder(h)
        return self.output_conv(h)
    
    def forward(self, x: 'torch.Tensor') -> VAEOutput:
        """Forward pass through VAE.
        
        Parameters
        ----------
        x : torch.Tensor
            Input field of shape (batch, channels, height, width).
            
        Returns
        -------
        output : VAEOutput
            Named tuple with reconstruction, mu, log_var, and z.
        """
        if not _TORCH_AVAILABLE:
            return VAEOutput(None, None, None, None)
        
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        
        return VAEOutput(reconstruction=reconstruction, mu=mu, log_var=log_var, z=z)
    
    def loss_function(
        self,
        reconstruction: 'torch.Tensor',
        original: 'torch.Tensor',
        mu: 'torch.Tensor',
        log_var: 'torch.Tensor',
    ) -> Dict[str, 'torch.Tensor']:
        """Compute VAE loss.
        
        Parameters
        ----------
        reconstruction : torch.Tensor
            Reconstructed field.
        original : torch.Tensor
            Original input field.
        mu : torch.Tensor
            Latent mean.
        log_var : torch.Tensor
            Latent log variance.
            
        Returns
        -------
        losses : Dict[str, torch.Tensor]
            Dictionary with 'loss', 'recon_loss', and 'kl_loss'.
        """
        if not _TORCH_AVAILABLE:
            return {'loss': 0, 'recon_loss': 0, 'kl_loss': 0}
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, original, reduction='sum') / original.size(0)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / original.size(0)
        
        # Total loss with beta weighting
        loss = recon_loss + self.beta * kl_loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
        }
    
    def fit(
        self,
        train_data: Union[np.ndarray, 'torch.Tensor'],
        val_data: Optional[Union[np.ndarray, 'torch.Tensor']] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> TrainingResult:
        """Train the VAE on field data.
        
        Parameters
        ----------
        train_data : array-like
            Training fields of shape (n_samples, height, width) or
            (n_samples, channels, height, width).
        val_data : array-like, optional
            Validation fields.
        epochs : int
            Number of training epochs.
        batch_size : int
            Batch size for training.
        learning_rate : float
            Learning rate for Adam optimizer.
        device : str, optional
            Device to train on ('cuda' or 'cpu'). Auto-detected if None.
        early_stopping_patience : int
            Stop training if validation loss doesn't improve for this many epochs.
        verbose : bool
            Whether to print training progress.
            
        Returns
        -------
        result : TrainingResult
            Training results with loss history.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for training. Install with: pip install torch")
        
        # Setup device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.to(device)
        
        # Prepare data
        train_tensor = self._prepare_data(train_data, device)
        val_tensor = self._prepare_data(val_data, device) if val_data is not None else None
        
        # Create data loader
        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        train_losses = []
        val_losses = [] if val_tensor is not None else None
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.train()
            epoch_loss = 0.0
            
            for (batch,) in train_loader:
                optimizer.zero_grad()
                output = self(batch)
                losses = self.loss_function(output.reconstruction, batch, output.mu, output.log_var)
                losses['loss'].backward()
                optimizer.step()
                epoch_loss += losses['loss'].item() * batch.size(0)
            
            epoch_loss /= len(train_tensor)
            train_losses.append(epoch_loss)
            
            # Validation
            if val_tensor is not None:
                self.eval()
                with torch.no_grad():
                    output = self(val_tensor)
                    val_loss_dict = self.loss_function(
                        output.reconstruction, val_tensor, output.mu, output.log_var
                    )
                    val_loss = val_loss_dict['loss'].item()
                    val_losses.append(val_loss)
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}")
        
        return TrainingResult(
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=best_epoch,
            final_loss=train_losses[-1],
        )
    
    def _prepare_data(self, data: Union[np.ndarray, 'torch.Tensor'], device: str) -> 'torch.Tensor':
        """Prepare data tensor for training."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        
        # Add channel dimension if needed
        if data.ndim == 3:
            data = data.unsqueeze(1)
        
        return data.to(device)
    
    def encode_fields(self, fields: Union[np.ndarray, 'torch.Tensor']) -> np.ndarray:
        """Encode fields to latent space.
        
        Parameters
        ----------
        fields : array-like
            Fields to encode, shape (n_samples, height, width).
            
        Returns
        -------
        latent : np.ndarray
            Latent representations, shape (n_samples, latent_dim).
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required")
        
        self.eval()
        device = next(self.parameters()).device
        
        tensor = self._prepare_data(fields, str(device))
        
        with torch.no_grad():
            mu, _ = self.encode(tensor)
        
        return mu.cpu().numpy()
    
    def decode_latent(self, z: Union[np.ndarray, 'torch.Tensor']) -> np.ndarray:
        """Decode latent vectors to fields.
        
        Parameters
        ----------
        z : array-like
            Latent vectors, shape (n_samples, latent_dim).
            
        Returns
        -------
        fields : np.ndarray
            Reconstructed fields, shape (n_samples, height, width).
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required")
        
        self.eval()
        device = next(self.parameters()).device
        
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()
        z = z.to(device)
        
        with torch.no_grad():
            recon = self.decode(z)
        
        return recon.squeeze(1).cpu().numpy()
    
    def reconstruct(self, fields: Union[np.ndarray, 'torch.Tensor']) -> np.ndarray:
        """Reconstruct fields through the VAE.
        
        Parameters
        ----------
        fields : array-like
            Input fields, shape (n_samples, height, width).
            
        Returns
        -------
        reconstructions : np.ndarray
            Reconstructed fields, shape (n_samples, height, width).
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required")
        
        self.eval()
        device = next(self.parameters()).device
        
        tensor = self._prepare_data(fields, str(device))
        
        with torch.no_grad():
            output = self(tensor)
        
        return output.reconstruction.squeeze(1).cpu().numpy()
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate random field samples from the prior.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
            
        Returns
        -------
        samples : np.ndarray
            Generated fields, shape (n_samples, height, width).
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required")
        
        self.eval()
        device = next(self.parameters()).device
        
        z = torch.randn(n_samples, self.latent_dim).to(device)
        
        with torch.no_grad():
            samples = self.decode(z)
        
        return samples.squeeze(1).cpu().numpy()
    
    def interpolate(
        self,
        field1: np.ndarray,
        field2: np.ndarray,
        n_steps: int = 10,
    ) -> np.ndarray:
        """Interpolate between two fields in latent space.
        
        Parameters
        ----------
        field1 : np.ndarray
            First field, shape (height, width).
        field2 : np.ndarray
            Second field, shape (height, width).
        n_steps : int
            Number of interpolation steps.
            
        Returns
        -------
        interpolation : np.ndarray
            Interpolated fields, shape (n_steps, height, width).
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required")
        
        # Encode both fields
        z1 = self.encode_fields(field1[np.newaxis, ...])[0]
        z2 = self.encode_fields(field2[np.newaxis, ...])[0]
        
        # Linear interpolation in latent space
        alphas = np.linspace(0, 1, n_steps)
        z_interp = np.array([z1 * (1 - a) + z2 * a for a in alphas])
        
        # Decode interpolated latents
        return self.decode_latent(z_interp)


# Convenience function for quick VAE creation
def create_field_vae(
    input_shape: Tuple[int, int],
    latent_dim: int = 32,
    architecture: str = "standard",
) -> FieldAutoencoder:
    """Create a FieldAutoencoder with sensible defaults.
    
    Parameters
    ----------
    input_shape : Tuple[int, int]
        Field dimensions (height, width).
    latent_dim : int
        Latent space dimension.
    architecture : str
        'standard' or 'deep'.
        
    Returns
    -------
    vae : FieldAutoencoder
        Configured VAE ready for training.
    """
    return FieldAutoencoder(
        input_shape=input_shape,
        latent_dim=latent_dim,
        architecture=architecture,
        base_channels=32,
        beta=1.0,
    )
